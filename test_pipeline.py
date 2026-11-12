"""Unit tests of circuit semantics and logical scoring; all execution is local."""
import unittest
import copy
import json
from pathlib import Path
import numpy as np
import stim
from qiskit import transpile
from qiskit_aer import AerSimulator
from pipeline import to_qiskit, measurement_array, benchmark, generated_memory, qubit_mapping, with_noise_probability


class PipelineTests(unittest.TestCase):
    """Check translation semantics, measurement order, surface-code resources, and decoder scoring."""
    def test_si1000_samples_keep_final_boundary_and_disjoint_splits(self):
        """Validate source-based sampling, final-boundary alignment, integrity and repeatability."""
        from sweep import create_si1000_dataset, si1000_training_arrays
        path = Path(__file__).resolve().parent/'runs/surface-d7-baseline/sweep/benchmark.json'
        if not path.is_file():
            self.skipTest('Requires the optional local SI1000 source archive')
        published = json.loads(path.read_text(encoding='utf-8'))['si1000_published_sources']
        document = {'si1000_published_sources': published}
        first = create_si1000_dataset(document, 'test', [3], ['z'], [.001], [16, 8, 8], 123)
        case = first['cases'][0]
        arrays = si1000_training_arrays(case, 'test')
        self.assertEqual(arrays['inputs']['ancilla_measurements'].shape, (8, 3, 8))
        self.assertEqual(arrays['inputs']['detector_events'].shape, (8, 4, 8))
        self.assertEqual(arrays['inputs']['detector_mask'].sum(), 24)
        self.assertEqual(arrays['inputs']['detector_mask'][-1].sum(), 4)
        self.assertEqual(arrays['labels'].shape, (8, 1))
        self.assertNotIn('labels', arrays['inputs'])
        self.assertEqual(len({s['seed'] for s in case['splits'].values()}), 3)
        rerun = create_si1000_dataset({'si1000_published_sources': published}, 'test', [3], ['z'], [.001], [16, 8, 8], 123)
        self.assertEqual(case['splits'], rerun['cases'][0]['splits'])
        # Enforce dataset immutability and verify that checksum errors cannot silently enter training.
        with self.assertRaisesRegex(ValueError, 'already exists'):
            create_si1000_dataset(document, 'test', [3], ['z'], [.001], [16, 8, 8], 123)
        corrupted = copy.deepcopy(case)
        corrupted['splits']['test']['labels_sha256'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'label checksum'):
            si1000_training_arrays(corrupted, 'test')

    def test_planar_grid_geometry_matches_css_stabilizers(self):
        """Both planar drawings must use real commuting checks and their actual boundary supports."""
        from compare_hardware_shots import qec_grid
        for layout,data,checks,weights in [('rotated',9,8,{2,4}),('unrotated',13,12,{3,4})]:
            models,geometry=qec_grid(layout,3)
            self.assertEqual(len(models['z']['qubits']),data)
            self.assertEqual(len(geometry['plaquettes']),checks)
            self.assertEqual({p['weight'] for p in geometry['plaquettes']},weights)
            self.assertFalse(np.any((models['x']['matrix']@models['z']['matrix'].T)%2))
            self.assertEqual(geometry['logical_qubits'],1)
            self.assertIsNone(geometry['period'])
            for p in geometry['plaquettes']:
                self.assertEqual(len(p['support']),p['weight'])
                self.assertGreaterEqual(len(p['vertices']),3)

    def test_toric_grid_has_periodic_checks_and_two_logical_qubits(self):
        """The torus decoder must correct local errors but retain noncontractible logical loops."""
        from compare_hardware_shots import qec_grid,qec_cycle,simulate_qec
        models,geometry=qec_grid('toric',3);zm,xm=models['z'],models['x']
        self.assertEqual(len(zm['qubits']),18)
        self.assertEqual(len(geometry['plaquettes']),18)
        self.assertEqual(geometry['logical_qubits'],2)
        self.assertEqual(geometry['period'],6)
        self.assertFalse(np.any((xm['matrix']@zm['matrix'].T)%2))
        self.assertFalse(np.any((xm['matrix']@zm['logical'].T)%2))
        self.assertFalse(np.any((zm['matrix']@xm['logical'].T)%2))
        np.testing.assert_array_equal((xm['logical']@zm['logical'].T)%2,np.eye(2,dtype=int))
        self.assertFalse(np.bitwise_xor.reduce(xm['matrix'],axis=0).any())
        self.assertFalse(np.bitwise_xor.reduce(zm['matrix'],axis=0).any())
        zero=np.zeros(18,dtype=np.uint8)
        for q in range(18):
            error=zero.copy();error[q]=1
            for nx,nz in [(error,zero),(zero,error),(error,error)]:
                cycle=qec_cycle(zero,zero,nx,nz,zm,xm)
                self.assertEqual(cycle['remaining_checks'],0)
                self.assertEqual(cycle['logical'],[0,0,0,0])
        loop=qec_cycle(zero,zero,xm['logical'][0],zero,zm,xm)
        self.assertEqual(loop['remaining_checks'],0)
        self.assertEqual(loop['logical'],[1,0,0,0])
        self.assertFalse(loop['correction_x'].any())
        clean=simulate_qec(0,3,2,42,3,'toric')
        self.assertEqual(clean['summary']['failures_by_round'],[0,0,0])
        self.assertEqual(len(clean['samples'][0][0]['logical']),4)

    def test_active_3d_trajectory_matches_actual_stim_checks(self):
        """Animated ideal check bits and final logical-Z readout must match the realized quantum circuit."""
        from compare_hardware_shots import active_memory_context,active_memory_shot,simulate_qec,build_tqec_model
        ctx = active_memory_context(3)
        base = simulate_qec(.05,3,3,9,3)
        batch = {'context':ctx,'active':base,'model':build_tqec_model('memory',True),
                 'parameters':{'model':'memory','mode':'active','k':1,'noise':.05,'rounds':3,'shots':3,'seed':9}}
        for index in range(3):
            result = active_memory_shot(batch,index)
            circuit = stim.Circuit(result['circuit'])
            bits = circuit.compile_sampler(seed=42).sample(1)[0]
            count = len(result['expected_checks'])
            np.testing.assert_array_equal(bits[:count],result['expected_checks'])
            logical = int(ctx['models']['z']['logical']@bits[count:].astype(np.uint8))%2
            self.assertEqual(logical,result['after'][0])
            self.assertEqual(len([f for f in result['active_frames'] if f['stage']=='after']),3)
            self.assertTrue(all(not f['correction'] for f in result['active_frames'] if f['stage'] in {'noise','extract','syndrome'}))

    def test_active_3d_y_fault_propagates_then_is_removed(self):
        """A central Y must spread through extraction gates, fire four checks, and vanish after recovery."""
        from compare_hardware_shots import active_memory_context,active_memory_shot,simulate_qec,build_tqec_model,qec_cycle,pauli_frame_map
        ctx = active_memory_context(7)
        base = simulate_qec(0,1,1,42,7)
        zm,xm = ctx['models']['z'],ctx['models']['x']
        zero = np.zeros(49,dtype=np.uint8); noise=zero.copy();noise[24]=1
        cycle=qec_cycle(zero,zero,noise,noise,zm,xm)
        history=base['samples'][0][0]
        history.update(noise=pauli_frame_map(zm['qubits'],noise,noise),
            before=pauli_frame_map(zm['qubits'],cycle['before_x'],cycle['before_z']),
            correction=pauli_frame_map(zm['qubits'],cycle['correction_x'],cycle['correction_z']),
            after=pauli_frame_map(zm['qubits'],cycle['after_x'],cycle['after_z']),logical=cycle['logical'])
        for check in history['checks']:
            model,values=(zm,cycle['z_checks']) if check['basis']=='Z' else (xm,cycle['x_checks'])
            node=next(n for n in ctx['geometry']['nodes'] if n['id']==check['qubit'])
            row=model['check_coordinates'].index([node['x'],node['y']])
            check['bit']=int(values[row])
        batch={'context':ctx,'active':base,'model':build_tqec_model('memory',True),
               'parameters':{'model':'memory','mode':'active','k':3,'noise':0,'rounds':1,'shots':1,'seed':42}}
        frames=active_memory_shot(batch,0)['active_frames']
        self.assertGreater(max(len(f['active']) for f in frames),1)
        self.assertTrue(any(f['transfers'] for f in frames))
        before=next(f for f in frames if f['stage']=='before')
        self.assertEqual(sum(c['bit'] for c in before['checks']),4)
        self.assertTrue(before['correction'])
        self.assertFalse(frames[-1]['active'])
        self.assertEqual(frames[-1]['logical'],[0,0])

    def test_explicit_noise_truth_and_readout_flips(self):
        """A known physical X followed by a known readout flip must cancel in the measured record."""
        from compare_hardware_shots import realize_stim_noise, sample_tqec_circuit
        circuit = stim.Circuit('QUBIT_COORDS(0,0) 0\nR 0\nX_ERROR(1) 0\nTICK\nM(1) 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]')
        realized,events,flips = realize_stim_noise(circuit,np.random.default_rng(5))
        bits = realized.compile_sampler(seed=1).sample(1)[0]
        self.assertEqual(int(bits[0]),1)
        bits[flips] ^= True
        self.assertEqual(int(bits[0]),0)
        self.assertEqual([e['pauli'] for e in events],['X','readout'])
        result = sample_tqec_circuit(circuit,shots=2,seed=5)
        self.assertFalse(result['detectors'].any())
        self.assertFalse(result['observables'].any())
        self.assertEqual(result['summary']['after_failures'],0)

    def test_tqec_compilation_and_simulated_recovery_are_connected(self):
        """The displayed circuit, exact samples and 3D graph recovery must describe one compiled model."""
        from tqec import NoiseModel
        from compare_hardware_shots import compile_tqec_circuit, sample_tqec_circuit, tqec_simulation_shot
        compiled = compile_tqec_circuit('memory',1)
        circuit = NoiseModel.uniform_depolarizing(.01).noisy_circuit(compiled['ideal'])
        sample = sample_tqec_circuit(circuit,shots=2,seed=7)
        batch = {'circuit':circuit,'samples':sample,'model':compiled['model'],
                 'parameters':{'model':'memory','k':1,'noise':.01,'shots':2,'seed':7}}
        shot = tqec_simulation_shot(batch,0)
        self.assertGreater(shot['resources']['detectors'],0)
        self.assertEqual(shot['remaining_graph_syndrome'],0)
        np.testing.assert_array_equal(shot['after'],np.array(shot['before'])^np.array(shot['prediction']))
        self.assertEqual(stim.Circuit(shot['circuit']),circuit)
        self.assertIn('<svg',shot['circuit_svg'])
        for event in shot['faults']:
            self.assertIn(event['qubit'],circuit.get_final_qubit_coordinates())
            self.assertTrue(all(0<=v<=1 for v in event['position']))
        with self.assertRaises(ValueError):
            tqec_simulation_shot(batch,99)

    def test_tqec_models_are_library_validated(self):
        """Displayed graph counts and port labels must come from real validated TQEC examples."""
        from compare_hardware_shots import build_tqec_model
        for name,cubes,pipes,surfaces in [('memory',1,0,1),('cnot',10,9,4),('three_cnots',12,12,6)]:
            model = build_tqec_model(name)
            metadata = model['metadata']
            self.assertTrue(metadata['validated'])
            self.assertEqual(len(metadata['cubes']),cubes)
            self.assertEqual(len(metadata['pipes']),pipes)
            self.assertEqual(metadata['correlation_surfaces'],surfaces)
            for pipe in metadata['pipes']:
                self.assertLess(pipe['u'],cubes)
                self.assertLess(pipe['v'],cubes)
        with self.assertRaises(ValueError):
            build_tqec_model('unknown')

    def test_tqec_native_viewer_uses_local_pinned_assets(self):
        """Surface and cutaway options retain native geometry without writing HTML/DAE files."""
        from compare_hardware_shots import build_tqec_model, tqec_native_html, fetch_tqec_asset
        model = build_tqec_model('cnot')
        plain = tqec_native_html(model)
        surface = tqec_native_html(model,0,True)
        self.assertIn('ColladaLoader',plain)
        self.assertIn('/api/tqec/asset?name=three',plain)
        self.assertNotIn('https://unpkg.com',plain)
        self.assertNotEqual(plain,surface)
        self.assertIn('tqec-ready',surface)
        with self.assertRaises(ValueError):
            tqec_native_html(model,99)
        with self.assertRaises(ValueError):
            fetch_tqec_asset('arbitrary-url')

    def test_qec_cycle_uses_syndrome_and_applies_recovery(self):
        """Ideal per-round recovery handles single Paulis without using hidden logical truth."""
        from compare_hardware_shots import final_recovery_model, qec_cycle
        zm = final_recovery_model(generated_memory(distance=7,rounds=1,basis='z',noise=0))
        xm = final_recovery_model(generated_memory(distance=7,rounds=1,basis='x',noise=0))
        zero = np.zeros(49,dtype=np.uint8)
        self.assertFalse(np.any((zm['matrix']@xm['matrix'].T)%2))
        for pauli in ['X','Y','Z']:
            nx,nz = zero.copy(),zero.copy()
            nx[24],nz[24] = int(pauli in 'XY'),int(pauli in 'YZ')
            cycle = qec_cycle(zero,zero,nx,nz,zm,xm)
            self.assertEqual(cycle['remaining_checks'],0)
            self.assertEqual(cycle['logical'],[0,0])
            next_round = qec_cycle(cycle['after_x'],cycle['after_z'],zero,zero,zm,xm)
            self.assertFalse(next_round['correction_x'].any())
            self.assertFalse(next_round['correction_z'].any())
        # An undetectable logical string must survive; the decoder cannot see truth.
        logical = qec_cycle(zero,zero,zm['logical_string'],zero,zm,xm)
        self.assertFalse(logical['correction_x'].any())
        self.assertEqual(logical['remaining_checks'],0)
        self.assertEqual(logical['logical'],[1,0])

    def test_qec_batch_is_reproducible_and_zero_noise_stays_clean(self):
        """Fresh QEC batches preserve round states, paired noise streams, and bounded inputs."""
        from compare_hardware_shots import simulate_qec
        clean = simulate_qec(noise=0,rounds=3,shots=2,seed=42)
        self.assertEqual(clean['summary']['failures_by_round'],[0,0,0])
        self.assertEqual(clean['summary']['uncorrected_failures_by_round'],[0,0,0])
        for history in clean['samples']:
            for frame in history:
                self.assertFalse(frame['before'])
                self.assertFalse(frame['correction'])
                self.assertFalse(frame['after'])
        a = simulate_qec(noise=.05,rounds=3,shots=2,seed=9)
        b = simulate_qec(noise=.05,rounds=3,shots=2,seed=9)
        self.assertEqual(a['samples'],b['samples'])
        self.assertGreater(a['summary']['injected_paulis'],0)
        for history in a['samples']:
            for frame in history:
                self.assertEqual(frame['remaining_checks'],0)
                self.assertEqual(len(frame['checks']),48)
        with self.assertRaises(ValueError):
            simulate_qec(rounds=15,shots=500)

    def test_noise_prior_change_preserves_ideal_circuit(self):
        """Changing p must alter decoder assumptions without silently changing hardware gates."""
        c = generated_memory(noise=0.005)
        changed = with_noise_probability(c,0.01)
        self.assertEqual(changed.without_noise(),c.without_noise().flattened())
        for op in changed:
            if op.name in {'DEPOLARIZE1','DEPOLARIZE2','X_ERROR'}:
                self.assertEqual(op.gate_args_copy(),[0.01])

    def test_counterfactual_pauli_trace_matches_known_fault(self):
        """The sandbox must match a deliberately injected Clifford fault, not invent hardware faults."""
        from compare_hardware_shots import pauli_trace
        prefix='QUBIT_COORDS(0,0) 0\nQUBIT_COORDS(1,0) 1\nR 0 1\nTICK\n'
        suffix='CX 0 1\nTICK\nMR 1\nDETECTOR rec[-1]\nTICK\nM 0\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]'
        c=stim.Circuit(prefix+suffix)
        for pauli in ['X','Y','Z']:
            trace=pauli_trace(c,0,pauli,1)
            injected=stim.Circuit(prefix+f'{pauli} 0\n'+suffix)
            m=injected.compile_sampler(seed=42).sample(1)
            det,obs=c.compile_m2d_converter().convert(measurements=m,separate_observables=True)
            self.assertEqual(trace['fired_detectors'],np.flatnonzero(det[0]).tolist())
            self.assertEqual(trace['logical_flip'],obs[0].astype(int).tolist())
            self.assertNotIn(1,trace['frames'][-1]['active'])
            # An event appears when its measurement exists, never in earlier gate layers.
            emitted = [i for frame in trace['frames'] for i in frame['new_detectors']]
            self.assertEqual(sorted(emitted),trace['fired_detectors'])
            self.assertFalse(trace['frames'][1]['new_detectors'])
            if pauli in {'X','Y'}:
                self.assertIn({'source':0,'target':1,'pauli':'X'},trace['frames'][1]['transfers'])
                self.assertEqual(trace['frames'][2]['new_detectors'],[0])

    def test_predicted_final_frame_preserves_measured_data_and_logical_class(self):
        """A representative recovery must satisfy final checks and retain the requested logical class."""
        from compare_hardware_shots import final_recovery_model, predicted_final_recovery
        for basis,pauli in [('z','X'),('x','Z')]:
            circuit = generated_memory(distance=3,rounds=3,basis=basis,noise=0)
            model = final_recovery_model(circuit)
            measured = circuit.compile_sampler(seed=5).sample(1)[0]
            measured[model['start']] ^= True
            original = measured.copy()
            _,obs = circuit.compile_m2d_converter().convert(measurements=measured.reshape(1,-1),separate_observables=True)
            for logical_prediction in [0,1]:
                recovery = predicted_final_recovery(model,measured,[logical_prediction])
                self.assertEqual(recovery['pauli'],pauli)
                self.assertEqual(recovery['logical_before'],int(obs[0,0]))
                self.assertEqual(recovery['logical_after'],int(obs[0,0])^logical_prediction)
                self.assertEqual(recovery['final_checks_after'],0)
                correction = np.array([r['flip'] for r in recovery['data']],dtype=np.uint8)
                self.assertEqual(int(model['logical']@correction)%2,logical_prediction)
                np.testing.assert_array_equal(measured,original)
            # A deliberately wrong logical prediction can satisfy all spatial checks.
            wrong = predicted_final_recovery(model,measured,[1-int(obs[0,0])])
            self.assertEqual(wrong['final_checks_after'],0)
            self.assertEqual(wrong['logical_after'],1)

    def test_sparse_qubit_labels(self):
        """Ensure sparse Stim indices compact to the correct active qubits and preserve outcomes."""
        c = stim.Circuit('R 10 200\nX 10\nCX 10 200\nM 10 200')
        qc = to_qiskit(c)
        self.assertEqual(qc.num_qubits, 2)
        self.assertEqual(qc.metadata['stim_to_qiskit_qubits'], {10: 0, 200: 1})
        sim = AerSimulator(method='stabilizer')
        r = sim.run(transpile(qc, sim), shots=16, memory=True).result()
        self.assertTrue(measurement_array(r.get_memory(), 2).all())

    def test_surface_d7_resources(self):
        """Verify the chosen patch uses 97 qubits and the expected measurement and detector counts."""
        for basis in ['x', 'z']:
            c = generated_memory(basis=basis)
            self.assertEqual(len(qubit_mapping(c)), 97)
            self.assertEqual(c.num_measurements, 48*7+49)
            self.assertEqual(c.num_detectors, 336)
            self.assertEqual(to_qiskit(c).num_qubits, 97)

    def test_noiseless_surface_benchmark(self):
        """Check zero-noise logical scoring, including a graph with no modeled faults."""
        c = generated_memory(noise=0)
        m = c.compile_sampler(seed=42).sample(32)
        self.assertEqual(benchmark(c, m)['failures'], 0)

    def test_surface_both_memory_bases(self):
        """Check ideal detector relationships for both X- and Z-basis surface-code memory."""
        sim = AerSimulator(method='stabilizer')
        for basis in ['x', 'z']:
            c = generated_memory(distance=3, rounds=3, basis=basis)
            result = sim.run(transpile(to_qiskit(c), sim), shots=64,
                             memory=True, seed_simulator=42).result()
            m = measurement_array(result.get_memory(), c.num_measurements)
            dets, obs = c.compile_m2d_converter().convert(
                measurements=m, separate_observables=True)
            self.assertFalse(dets.any())
            self.assertFalse(obs.any())

    def test_bit_order(self):
        """Verify Qiskit display order is reversed into chronological Stim record order."""
        np.testing.assert_array_equal(measurement_array(['100', '011'], 3),
                                      [[0, 0, 1], [1, 1, 0]])

    def test_translation_measurement_parities(self):
        # Nonzero records and random outcomes exercise reference samples and ordering.
        """Exercise nonzero reference outcomes and random measurements without spurious events."""
        c = stim.Circuit('R 0 1\nX 0\nM 0\nH 1\nM 1\nM 0\n'
                         'DETECTOR rec[-3] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]')
        sim = AerSimulator(method='stabilizer')
        result = sim.run(transpile(to_qiskit(c), sim), shots=128,
                         memory=True, seed_simulator=1).result()
        m = measurement_array(result.get_memory(), c.num_measurements)
        dets, obs = c.compile_m2d_converter().convert(measurements=m, separate_observables=True)
        self.assertFalse(dets.any())
        self.assertFalse(obs.any())
        self.assertTrue(m[:, 0].all())
        self.assertGreater(m[:, 1].sum(), 0)
        self.assertLess(m[:, 1].sum(), len(m))

    def test_repetition_translation_and_decoder(self):
        """Check ideal translation and useful decoding on reproducible noisy repetition shots."""
        c = stim.Circuit.generated('repetition_code:memory', distance=3, rounds=3,
                                  after_clifford_depolarization=0.01,
                                  before_measure_flip_probability=0.01)
        sim = AerSimulator(method='stabilizer')
        r = sim.run(transpile(to_qiskit(c), sim), shots=128, memory=True).result()
        self.assertEqual(benchmark(c, measurement_array(r.get_memory(), c.num_measurements))['failures'], 0)
        m = c.compile_sampler(seed=1).sample(10000)
        decoded = benchmark(c, m)
        raw = benchmark(c, m, 'none')
        self.assertLess(decoded['failures'], raw['failures'])

    def test_unsupported_fails(self):
        """Reject operations whose semantics the converter cannot safely preserve."""
        for text in ['MPP X0*X1', 'M !0', 'M 0\nCX rec[-1] 1']:
            with self.subTest(circuit=text), self.assertRaises(ValueError):
                to_qiskit(stim.Circuit(text))

    def test_measure_reset_axes(self):
        """Check X/Y basis measurement and reset prepare the expected eigenstates."""
        c = stim.Circuit('RX 0\nMX 0\nMRX 0\nMX 0\nRY 1\nMY 1\nMRY 1\nMY 1')
        sim = AerSimulator(method='stabilizer')
        r = sim.run(transpile(to_qiskit(c), sim), shots=128, memory=True).result()
        m = measurement_array(r.get_memory(), c.num_measurements)
        self.assertFalse(m.any())


if __name__ == '__main__':
    unittest.main()
