"""Run bounded IBM trials; append-mode preserves existing files and older shots.

Normal phases keep the original directory-based workflow. The *-extra phases
store additional jobs and compressed shot records inside existing JSON files.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zlib
from datetime import datetime, timezone

import numpy as np
import stim
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from pipeline import benchmark, generated_memory, load, measurement_array, service, to_qiskit, versions, write_json, with_noise_probability


def encode_measurements(measurements):
    """Serialize an array losslessly in memory so no extra .npy file is needed."""
    buffer = io.BytesIO()
    np.save(buffer, measurements, allow_pickle=False)
    raw = buffer.getvalue()
    return {'encoding': 'npy-zlib-base64', 'sha256': hashlib.sha256(raw).hexdigest(),
            'data': base64.b64encode(zlib.compress(raw)).decode('ascii')}


def decode_measurements(payload):
    """Recover saved shots and check integrity without extracting a disk file."""
    if payload['encoding'] != 'npy-zlib-base64':
        raise ValueError('Unknown measurement encoding')
    raw = zlib.decompress(base64.b64decode(payload['data']))
    if hashlib.sha256(raw).hexdigest() != payload['sha256']:
        raise ValueError('Saved measurement checksum mismatch')
    return np.load(io.BytesIO(raw), allow_pickle=False)


def single_qubit_calibration(gates):
    """Summarize reported 1Q gate errors without mixing gate types or readout."""
    by_gate = {}
    for row in gates:
        name = row['gate']
        if len(row['qubits']) != 1 or name.startswith(('measure', 'reset')) or row['error'] is None:
            continue
        by_gate.setdefault(name, []).append(row['error'])
    return {name: {'median_error': float(np.median(values)), 'qubit_count': len(values)}
            for name, values in sorted(by_gate.items())}


def hardware_calibration(job):
    """Retrieve provider calibration at execution time, not hidden per-shot fault labels."""
    snapshot = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'IBM job-execution calibration estimates',
        'pauli_fault_rates': {'X': None, 'Y': None, 'Z': None},
        'limitations': 'Syndrome records do not identify actual X/Y/Z faults, leakage, or a unique decoherence history.'}
    try:
        metrics = job.metrics() or {}
        snapshot['job_metrics'] = {key: metrics[key] for key in
            ('timestamps', 'usage', 'qiskit_version', 'executions') if key in metrics}
    except Exception:
        # Calibration and job accounting are independent optional endpoints.
        pass
    try:
        props = job.properties()
        if props is None:
            snapshot['unavailable'] = 'No execution-time calibration returned'
            return snapshot
        rows = []
        for q in range(len(props.qubits)):
            row = {'physical_qubit': q}
            for label, method, scale in [('t1_us', props.t1, 1e6), ('t2_us', props.t2, 1e6),
                                          ('readout_error', props.readout_error, 1)]:
                try:
                    row[label] = float(method(q))*scale
                except Exception:
                    row[label] = None
            rows.append(row)
            for key in ('init_error', 'prob_meas0_prep1', 'prob_meas1_prep0'):
                try:
                    row[key], updated = props.qubit_property(q, key)
                    row[key+'_updated_at'] = updated.isoformat()
                except Exception:
                    row.setdefault(key, None)
            try:
                row['readout_length_us'] = props.readout_length(q)*1e6
            except Exception:
                row['readout_length_us'] = None
        # Preserve IBM's reported units and dates rather than inferring a channel.
        def reported(parameter):
            return {'value': float(parameter.value), 'unit': getattr(parameter, 'unit', None),
                    'updated_at': str(getattr(parameter, 'date', None))}
        all_gates = [{'gate': g.gate, 'qubits': list(g.qubits),
                      'error': next((float(p.value) for p in g.parameters if p.name == 'gate_error'), None),
                      'properties': {p.name: reported(p) for p in g.parameters}}
                     for g in props.gates]
        gates = [g for g in all_gates if len(g['qubits']) == 2]
        snapshot.update(last_update=str(props.last_update_date), per_qubit=rows, two_qubit_gates=gates,
                        other_gates=[g for g in all_gates if len(g['qubits']) != 2],
                        single_qubit_gate_errors=single_qubit_calibration(all_gates))
        for label in ['t1_us', 't2_us', 'readout_error']:
            values = [r[label] for r in rows if r[label] is not None]
            snapshot['median_'+label] = float(np.median(values)) if values else None
        values = [g['error'] for g in gates if g['error'] is not None]
        snapshot['median_two_qubit_error'] = float(np.median(values)) if values else None
    except Exception as exc:
        # Preserve shot retrieval even when optional provider calibration is unavailable.
        snapshot['unavailable'] = f'{type(exc).__name__}: {exc}'
    return snapshot


def extra_experiment(args, out, manifest, svc):
    """Submit or collect an added job while modifying existing files only.

    The original manifest, benchmarks and measurements still describe their
    original job. Added experiments live under a separate dictionary key.
    """
    # Verify every destination exists before making a remote submission.
    comparison = Path(args.comparison)
    destinations = [out/'manifest.json', out/'status.json', out/'report.md',
                    comparison/'comparison.json', comparison/'report.md']
    destinations += [out/f'benchmark-{d}.json' for d in
                     ['none', 'pymatching', 'pymatching-correlated']]
    if any(not path.is_file() for path in destinations):
        raise ValueError('Append mode requires existing result and comparison files')
    # The existing ISA artifact fixes routing and physical-qubit placement.
    if hashlib.sha256((out/'isa.qpy').read_bytes()).hexdigest() != manifest['isa_sha256']:
        raise ValueError('Compiled circuit fingerprint mismatch')
    c = load(out/'circuit.stim')
    if hashlib.sha256(str(c).encode()).hexdigest() != manifest['circuit_sha256']:
        raise ValueError('Source circuit fingerprint mismatch')
    experiments = manifest.setdefault('additional_experiments', {})
    key = args.experiment
    if args.phase == 'submit-extra':
        # A saved submission record prevents a second job under the same key.
        if key in experiments:
            raise ValueError('Experiment key already exists; refusing duplicate submission')
        current = next(i for i in svc.instances() if i['crn'] == manifest['instance'])
        if current['plan'] != 'open':
            raise ValueError('Submission requires the free Open Plan')
        backend = svc.backend(manifest['backend'])
        if not backend.status().operational:
            raise ValueError('Selected backend is not operational')
        with (out/'isa.qpy').open('rb') as f:
            isa, = qpy.load(f)
        trial = {'shots': args.shots, 'backend': manifest['backend'],
                 'circuit_sha256': manifest['circuit_sha256'],
                 'isa_sha256': manifest['isa_sha256'],
                 'sampler_options': manifest['sampler_options'],
                 'status': 'submission_started', 'versions': versions()}
        if args.decoder_noise is not None:
            # Noise instructions are decoder assumptions; the submitted ISA is unchanged.
            model_circuit = with_noise_probability(c, args.decoder_noise)
            trial.update(decoder_noise=args.decoder_noise, decoder_circuit=str(model_circuit),
                         decoder_model_sha256=hashlib.sha256(str(model_circuit).encode()).hexdigest())
        # Persist intent first: an interrupted submission must not be retried blindly.
        experiments[key] = trial
        write_json(out/'manifest.json', manifest)
        sampler = SamplerV2(mode=backend, options=manifest['sampler_options'])
        job = sampler.run([isa], shots=args.shots)
        trial.update(job_id=job.job_id(), status='submitted')
        write_json(out/'manifest.json', manifest)
        with (out/'report.md').open('a', encoding='utf-8') as f:
            f.write(f"\n## Additional experiment: {key}\n\nSubmitted {args.shots} shots; job `{job.job_id()}`. Results pending.\n")
        print(f'Submitted exactly {args.shots} shots:', job.job_id(), flush=True)
        return
    # Collection never re-submits a job; repeated collection only refreshes data.
    trial = experiments[key]
    job = svc.job(trial['job_id'])
    result = job.result()[0]
    m = measurement_array(result.data.meas.get_bitstrings(), c.num_measurements)
    if len(m) != trial['shots']:
        raise ValueError('Returned shot count differs from the saved request')
    payload = encode_measurements(m)
    np.testing.assert_array_equal(decode_measurements(payload), m)
    dets, _ = c.compile_m2d_converter().convert(measurements=m, separate_observables=True)
    reports = {}
    model_circuit = stim.Circuit(trial['decoder_circuit']) if 'decoder_circuit' in trial else c
    for decoder in ['none', 'pymatching', 'pymatching-correlated']:
        # Each decoder sees the same frozen rows; logical labels are scoring-only.
        report = benchmark(model_circuit, m, decoder)
        report.update(source='ibm', job_id=trial['job_id'], backend=trial['backend'],
                      metadata=result.metadata, versions=versions())
        reports[decoder] = report
        path = out/f'benchmark-{decoder}.json'
        previous = json.loads(path.read_text())
        previous.setdefault('additional_experiments', {})[key] = report
        write_json(path, previous)
    status = {'status': job.status(), 'metrics': job.metrics()}
    calibration = hardware_calibration(job)
    previous_status = json.loads((out/'status.json').read_text())
    previous_status.setdefault('additional_experiments', {})[key] = status
    write_json(out/'status.json', previous_status)
    trial.update(status=status['status'], measurements=payload, calibration=calibration)
    write_json(out/'manifest.json', manifest)
    # Extend the existing comparison; do not replace either original run.
    from compare_hardware_shots import append_trial
    append_trial(comparison, key, {'manifest': trial, 'results': reports,
        'status': status, 'measurements': payload,
        'calibration': calibration,
        'detector_events': int(dets.sum()), 'detector_opportunities': int(dets.size),
        'shots_with_detector_events': int(dets.any(axis=1).sum())})
    # Replace only this appended section so collection is safe to repeat.
    marker = f'\n## Additional experiment: {key}\n'
    text = (out/'report.md').read_text(encoding='utf-8-sig')
    text = text.split(marker)[0]
    text += marker + f"\nCompleted {len(m)} shots; job `{trial['job_id']}`.\n\n"
    text += '| Decoder | Logical failures | 95% Wilson interval |\n|---|---:|---|\n'
    for decoder, report in reports.items():
        lo, hi = report['wilson_95_interval']
        text += f"| {decoder} | {report['failures']}/{len(m)} | {100*lo:.1f}%–{100*hi:.1f}% |\n"
    text += '\nRaw measurements are preserved losslessly inside the existing manifest and comparison JSON.\n'
    (out/'report.md').write_text(text, encoding='utf-8')
    print(json.dumps({'job_id': trial['job_id'], 'shots': len(m),
        'failures': {d: r['failures'] for d, r in reports.items()},
        'detector_event_fraction': float(dets.mean()), 'metrics': status['metrics']}, indent=2), flush=True)


def memory_pilot(args, out, manifest, svc):
    """Submit or collect the two five-round d7 memory PUBs in one Open Plan job."""
    path = out/'manifest.json'
    pilots = manifest.setdefault('memory_pilots', {})
    key = args.experiment
    if args.phase in {'prepare-memory-pilot', 'submit-memory-pilot'}:
        if key in pilots:
            raise ValueError('Pilot key already exists; refusing duplicate submission')
        instance = next(i for i in svc.instances() if i['crn'] == manifest['instance'])
        if instance['plan'] != 'open' or args.shots != 10:
            raise ValueError('This pilot requires Open Plan and exactly ten shots per basis')
        backend = svc.backend(manifest['backend'])
        if backend.num_qubits < 97 or not backend.status().operational:
            raise ValueError('A live backend with at least 97 qubits is required')
        # Use the previous validated d7 placement for both bases.
        with (out/'isa.qpy').open('rb') as f:
            reference, = qpy.load(f)
        layout = reference.layout.initial_index_layout(filter_ancillas=True)
        pass_manager = generate_preset_pass_manager(backend=backend, initial_layout=layout,
                                                    optimization_level=1, seed_transpiler=42)
        sim = AerSimulator(method='stabilizer')
        pubs, cases = [], []
        for index, basis in enumerate(('x', 'z')):
            c = generated_memory(distance=7, rounds=5, noise=.01, basis=basis)
            isa = pass_manager.run(to_qiskit(c))
            ideal = transpile(isa, sim, optimization_level=0, seed_transpiler=42)
            result = sim.run(ideal, shots=10, memory=True, seed_simulator=42).result()
            bits = measurement_array(result.get_memory(), c.num_measurements)
            dets, obs = c.compile_m2d_converter().convert(measurements=bits, separate_observables=True)
            if dets.any() or obs.any():
                raise ValueError(f'Routed {basis.upper()} circuit failed ideal validation')
            blob = io.BytesIO()
            qpy.dump(isa, blob)
            pubs.append(isa)
            cases.append({'basis': basis, 'pub_index': index, 'shots': 10, 'rounds': 5,
                          'distance': 7, 'decoder_noise': .01, 'circuit': str(c),
                          'isa_depth': isa.depth(), 'isa_operations': dict(isa.count_ops()),
                          'isa_qpy_base64': base64.b64encode(blob.getvalue()).decode('ascii'),
                          'ideal_validation_shots': 10})
            print(f'{basis.upper()}: ideal routed validation passed, depth={isa.depth()}, ops={dict(isa.count_ops())}', flush=True)
        if args.phase == 'prepare-memory-pilot':
            return
        pilot = {'backend': backend.name, 'cases': cases, 'status': 'submission_started',
                 'sampler_options': manifest['sampler_options'], 'versions': versions()}
        # Record submission intent before the remote action; collect never submits.
        pilots[key] = pilot
        write_json(path, manifest)
        job = SamplerV2(mode=backend, options=pilot['sampler_options']).run(pubs, shots=10)
        pilot.update(job_id=job.job_id(), status='submitted')
        write_json(path, manifest)
        print('Submitted 10 shots per basis:', job.job_id(), flush=True)
        return
    pilot = pilots[key]
    job = svc.job(pilot['job_id'])
    print(f"IBM job {job.job_id()}: {job.status()}", flush=True)
    result = job.result()
    pilot.update(status=job.status(), calibration=hardware_calibration(job))
    for case, pub in zip(pilot['cases'], result, strict=True):
        c = stim.Circuit(case['circuit'])
        bits = measurement_array(pub.data.meas.get_bitstrings(), c.num_measurements)
        if len(bits) != case['shots']:
            raise ValueError('Hardware pilot shot count differs from request')
        case.update(measurements=encode_measurements(bits),
                    results={d: benchmark(c, bits, d) for d in ('none', 'pymatching', 'pymatching-correlated')})
        print(case['basis'], {d:r['failures'] for d,r in case['results'].items()}, flush=True)
    write_json(path, manifest)


def ising_circuit(width, edges, time, coupling, field, steps):
    """Second-order product formula for H = -J sum ZZ - h sum X."""
    circuit = QuantumCircuit(width, width, name='ising')
    dt = time / steps
    for _ in range(steps):
        if field and time:
            circuit.rx(-field * dt, range(width))
        if coupling and time:
            for a, b in edges:
                circuit.rzz(-2 * coupling * dt, a, b)
        if field and time:
            circuit.rx(-field * dt, range(width))
    circuit.measure(range(width), range(width))
    return circuit


def ising_ideal_magnetization(width, edges, time, coupling, field, steps):
    """Calculate each ideal Z expectation using its small backward causal cone."""
    if not coupling or not time:
        return [float(np.cos(2 * field * time))] * width
    result = []
    dt = time / steps
    for site in range(width):
        support = {site}
        layers = []
        for _ in range(steps):
            # Within a commuting ZZ layer, only edges touching the incoming support matter.
            layer = [(a, b) for a, b in edges if a in support or b in support]
            layers.append(layer)
            support.update(q for edge in layer for q in edge)
        mapping = {q: i for i, q in enumerate(sorted(support))}
        cone = QuantumCircuit(len(mapping))
        for layer in reversed(layers):
            cone.rx(-field * dt, range(len(mapping)))
            for a, b in layer:
                cone.rzz(-2 * coupling * dt, mapping[a], mapping[b])
            cone.rx(-field * dt, range(len(mapping)))
        p = Statevector.from_instruction(cone).probabilities([mapping[site]])
        result.append(float(p[0] - p[1]))
    return result


def ising_trial(args, out, manifest, svc):
    """Execute a small, full-width native-connectivity Ising benchmark."""
    path = out/'manifest.json'
    experiments = manifest.setdefault('ising_experiments', {})
    key = args.experiment
    if args.phase == 'submit-ising':
        if key in experiments:
            raise ValueError('Experiment key already exists; refusing duplicate submission')
        instance = next(i for i in svc.instances() if i['crn'] == manifest['instance'])
        if instance['plan'] != 'open':
            raise ValueError('Ising trial requires the verified Open Plan')
        backend = svc.backend(manifest['backend'])
        width = backend.num_qubits
        if width != 156 or not backend.status().operational:
            raise ValueError('Choose an operational 156-qubit backend')
        props = backend.properties()
        edges = sorted({tuple(sorted(edge)) for edge in backend.target['cz']
                        if props.is_gate_operational('cz', edge)})
        if {q for edge in edges for q in edge} != set(range(width)) or props.faulty_qubits():
            raise ValueError('Not all 156 qubits have operational couplers')
        # Trivial placement plus disabled routing preserves every physical bond in the model.
        manager = generate_preset_pass_manager(backend=backend, optimization_level=1,
            layout_method='trivial', routing_method='none', seed_transpiler=42)
        options = {'twirling': {'enable_gates': False, 'enable_measure': False},
                   'execution': {'init_qubits': True}, 'max_execution_time': 60}
        design = [
            ('readout-control', 0., 0., 0., 1),
            ('field-only-control', .8, 0., .7, 1),
            ('ising-t04-one-step', .4, 1., .7, 1),
            ('ising-t08-one-step', .8, 1., .7, 1),
            ('ising-t08-two-steps', .8, 1., .7, 2),
        ]
        cases, pubs = [], []
        for label, t, coupling, field, steps in design:
            source = ising_circuit(width, edges, t, coupling, field, steps)
            isa = manager.run(source)
            measurement_order = {isa.find_bit(i.clbits[0]).index: isa.find_bit(i.qubits[0]).index
                                 for i in isa.data if i.operation.name == 'measure'}
            if measurement_order != {q: q for q in range(width)}:
                raise ValueError('Compiled measurements do not preserve physical qubit order')
            for instruction in isa.data:
                op = instruction.operation.name
                qubits = tuple(isa.find_bit(q).index for q in instruction.qubits)
                if op != 'barrier' and not backend.target.instruction_supported(operation_name=op, qargs=qubits):
                    raise ValueError(f'Unsupported compiled instruction: {op} on {qubits}')
            ideal = ising_ideal_magnetization(width, edges, t, coupling, field, steps)
            compiled = io.BytesIO()
            qpy.dump(isa, compiled)
            cases.append({'label': label, 'time': t, 'J': coupling, 'h': field, 'steps': steps,
                          'ideal_z_by_qubit': ideal, 'ideal_mean_z': float(np.mean(ideal)),
                          'isa_depth': isa.depth(), 'isa_operations': dict(isa.count_ops()),
                          'isa_qpy_base64': base64.b64encode(compiled.getvalue()).decode('ascii')})
            pubs.append(isa)
            print(label, 'CZ=', isa.count_ops().get('cz', 0), 'ideal mean Z=', round(np.mean(ideal), 6), flush=True)
        run = {'status': 'submission_started', 'backend': backend.name, 'num_qubits': width,
               'initial_state': 'all-zero product state', 'edges': edges, 'shots_per_case': args.shots,
               'sampler_options': options, 'versions': versions(), 'cases': cases}
        # Save intent before submitting. Collection can resume by key without spending more shots.
        experiments[key] = run
        write_json(path, manifest)
        job = SamplerV2(mode=backend, options=options).run(pubs, shots=args.shots)
        run.update(job_id=job.job_id(), status='submitted')
        write_json(path, manifest)
        print('Submitted Ising job', job.job_id(), flush=True)
        return
    run = experiments[key]
    job = svc.job(run['job_id'])
    print('Ising job', job.job_id(), job.status(), flush=True)
    results = job.result()
    # Reload after waiting so unrelated experiment records can be added concurrently.
    manifest = json.loads(path.read_text())
    run = manifest['ising_experiments'][key]
    edge_a, edge_b = np.array(run['edges']).T
    for case, pub in zip(run['cases'], results, strict=True):
        bits = measurement_array(pub.data.c.get_bitstrings(), run['num_qubits'])
        if len(bits) != run['shots_per_case']:
            raise ValueError('Unexpected shot count in Ising circuit')
        spins = 1 - 2*bits.astype(np.int8)
        z_shots = spins.mean(axis=1)
        zz_shots = (spins[:, edge_a] * spins[:, edge_b]).mean(axis=1)
        case.update(measurements=encode_measurements(bits), measured_z_by_qubit=spins.mean(axis=0).tolist(),
                    measured_mean_z=float(z_shots.mean()), mean_z_shot_se=float(z_shots.std(ddof=1)/np.sqrt(len(bits))),
                    measured_mean_zz=float(zz_shots.mean()), metadata=pub.metadata)
        print(case['label'], 'measured mean Z=', round(z_shots.mean(), 6),
              'measured mean ZZ=', round(zz_shots.mean(), 6), flush=True)
    metrics = job.metrics() or {}
    run.update(status=job.status(), job_metrics={k:metrics[k] for k in ('timestamps', 'usage') if k in metrics})
    write_json(path, manifest)


def with_ancilla_preparation(circuit, x_preparation):
    """Change the ancilla frame for each check and round, preserving record order."""
    circuit = circuit.without_noise().flattened()
    ancillas = {t.value for op in circuit if op.name == 'MR' for t in op.targets_copy()}
    out = stim.Circuit()
    round_index = 0
    hadamards = 0
    for op in circuit:
        if op.name == 'H':
            if not {t.value for t in op.targets_copy()} <= ancillas:
                raise ValueError('Expected the generated surface-code extraction schedule')
            # The existing H layers prepare and read out the ancilla in the X frame.
            chosen = sorted(x_preparation[round_index])
            if chosen:
                out.append('H', chosen)
            hadamards += 1
        elif op.name == 'CX':
            targets = [t.value for t in op.targets_copy()]
            chosen = x_preparation[round_index]
            for control, target in zip(targets[::2], targets[1::2]):
                if control in ancillas and control not in chosen:
                    # X parity in a Z ancilla frame: reverse the CNOT with data H gates.
                    out.append('H', [target])
                    out.append('CX', [target, control])
                    out.append('H', [target])
                elif target in ancillas and target in chosen:
                    # Z parity in an X ancilla frame: replace data-to-ancilla CX by CZ.
                    out.append('CZ', [control, target])
                else:
                    out.append('CX', [control, target])
        else:
            out.append(op)
            if op.name == 'MR':
                if hadamards != 2:
                    raise ValueError('Expected two ancilla H layers per extraction round')
                round_index += 1
                hadamards = 0
    if round_index != len(x_preparation):
        raise ValueError('Ancilla schedule length differs from the extraction circuit')
    return out


def ancilla_trial(args, out, manifest, svc):
    """Compare standard preparation with balanced randomized ancilla frames on IBM."""
    path = out/'manifest.json'
    experiments = manifest.setdefault('ancilla_experiments', {})
    key = args.experiment
    if args.phase in {'prepare-ancilla', 'submit-ancilla'}:
        saved = experiments.get(key)
        if saved is not None and (args.phase != 'submit-ancilla' or saved['status'] != 'prepared'):
            raise ValueError('Experiment key already exists; refusing duplicate submission')
        instance = next(i for i in svc.instances() if i['crn'] == manifest['instance'])
        if instance['plan'] != 'open':
            raise ValueError('This experiment requires the verified Open Plan')
        backend = svc.backend(manifest['backend'])
        if not backend.status().operational:
            raise ValueError('Backend is unavailable')
        if saved is not None:
            if saved['backend'] != backend.name or saved['distance'] != args.distance:
                raise ValueError('Prepared run differs from requested backend or distance')
            requested = args.total_shots if args.total_shots is not None else 12*args.shots
            if sum(case['shots'] for case in saved['cases']) != requested:
                raise ValueError('Prepared run differs from requested shot count')
            # Recover only circuits that already passed the compiled-circuit validation.
            pubs = [(qpy.load(io.BytesIO(base64.b64decode(case['isa_qpy_base64'])))[0], None, case['shots'])
                    for case in saved['cases']]
            saved['status'] = 'submission_started'
            write_json(path, manifest)
            job = SamplerV2(mode=backend, options=saved['sampler_options']).run(pubs)
            saved.update(job_id=job.job_id(), status='submitted')
            write_json(path, manifest)
            print('Submitted ancilla reset comparison', job.job_id(), flush=True)
            return
        options = {'twirling': {'enable_gates': False, 'enable_measure': False},
                   'execution': {'init_qubits': True},
                   'max_execution_time': 300 if (args.total_shots or 0) > 100000 else 60}
        seed = 147251 if args.total_shots is None else 386924 + args.distance
        rng = np.random.default_rng(seed)
        aer = AerSimulator(method='stabilizer', max_parallel_threads=1)
        layout = None
        cases, pubs = [], []
        pairs = 2 if args.total_shots is None else 4
        for round_index, rounds in enumerate((3, 5, 7)):
            template = generated_memory(distance=args.distance, rounds=rounds, basis='z', noise=0).flattened()
            ancillas = {t.value for op in template if op.name == 'MR' for t in op.targets_copy()}
            schedules = []
            # Each random schedule has a complement, with equal shots for both.
            for _ in range(pairs):
                schedule = [{q for q in sorted(ancillas) if rng.integers(2)} for _ in range(rounds)]
                schedules.extend((schedule, [ancillas - selected for selected in schedule]))
            for basis_index, basis in enumerate(('x', 'z')):
                if args.total_shots is None:
                    arm_shots = args.shots
                else:
                    # One unit pays for two shots in each arm, so complements remain balanced.
                    units, extra = divmod(args.total_shots // 4, 6)
                    arm_shots = 2*(units + (2*round_index+basis_index < extra))
                per_pair, extra_pairs = divmod(arm_shots//2, pairs)
                model = generated_memory(distance=args.distance, rounds=rounds, basis=basis, noise=.005)
                converter = model.compile_m2d_converter()
                for index, schedule in enumerate([None, *schedules]):
                    source = model.without_noise() if schedule is None else with_ancilla_preparation(model, schedule)
                    qc = to_qiskit(source)
                    manager = generate_preset_pass_manager(backend=backend, initial_layout=layout,
                        optimization_level=1, seed_transpiler=42)
                    isa = manager.run(qc)
                    if layout is None:
                        layout = isa.layout.initial_index_layout(filter_ancillas=True)
                    for inst in isa.data:
                        name = inst.operation.name
                        qubits = tuple(isa.find_bit(q).index for q in inst.qubits)
                        if name != 'barrier' and not backend.target.instruction_supported(operation_name=name, qargs=qubits):
                            raise ValueError(f'Unsupported compiled instruction: {name} on {qubits}')
                    ideal = transpile(isa, aer, optimization_level=0, seed_transpiler=42)
                    result = aer.run(ideal, shots=16, memory=True, seed_simulator=42).result()
                    bits = measurement_array(result.get_memory(), model.num_measurements)
                    detectors, observable = converter.convert(measurements=bits, separate_observables=True)
                    if detectors.any() or observable.any():
                        raise ValueError(f'Ancilla-frame variant failed ideal replay: {basis}, {rounds}, {index}')
                    blob = io.BytesIO()
                    qpy.dump(isa, blob)
                    try:
                        duration = isa.estimate_duration(backend.target, unit='s')
                    except Exception:
                        duration = None
                    shots = arm_shots if index == 0 else per_pair + ((index-1)//2 < extra_pairs)
                    cases.append({'basis': basis, 'rounds': rounds, 'variant': index, 'shots': shots,
                        'x_preparation_by_round': None if schedule is None else [sorted(s) for s in schedule],
                        'source_circuit': str(source), 'model_circuit': str(model),
                        'isa_qpy_base64': base64.b64encode(blob.getvalue()).decode('ascii'),
                        'duration_seconds': duration, 'isa_operations': dict(isa.count_ops())})
                    pubs.append((isa, None, shots))
            print(f'Validated distance {args.distance}, {rounds} rounds in both bases', flush=True)
        order = rng.permutation(len(pubs)).tolist()
        run = {'status': 'prepared' if args.phase == 'prepare-ancilla' else 'submission_started',
               'backend': backend.name, 'distance': args.distance, 'seed': seed,
               'initial_layout': layout, 'sampler_options': options, 'versions': versions(),
               'cases': [cases[i] for i in order]}
        print(f'Validated {len(pubs)} compiled circuits with 16 ideal shots each', flush=True)
        # Persist submission intent to guard against submitting twice after interruption.
        experiments[key] = run
        write_json(path, manifest)
        if args.phase == 'prepare-ancilla':
            print('Prepared', sum(case['shots'] for case in cases), 'shots for', key, flush=True)
            return
        job = SamplerV2(mode=backend, options=options).run([pubs[i] for i in order])
        run.update(job_id=job.job_id(), status='submitted')
        write_json(path, manifest)
        print('Submitted ancilla reset comparison', job.job_id(), flush=True)
        return

    job = svc.job(experiments[key]['job_id'])
    print('Ancilla reset job', job.job_id(), job.status(), flush=True)
    results = job.result()
    manifest = json.loads(path.read_text())
    run = manifest['ancilla_experiments'][key]
    groups = {}
    for case, pub in zip(run['cases'], results, strict=True):
        model = stim.Circuit(case['model_circuit'])
        bits = measurement_array(pub.data.meas.get_bitstrings(), model.num_measurements)
        if len(bits) != case['shots']:
            raise ValueError('Returned shot count does not match ancilla reset experiment')
        case['measurements'] = encode_measurements(bits)
        group = (case['basis'], case['rounds'], bool(case['variant']))
        groups.setdefault(group, []).append((model, bits))
    summary = []
    for (basis, rounds, randomized), rows in sorted(groups.items()):
        model = rows[0][0]
        bits = np.concatenate([bits for _, bits in rows])
        row = {'basis': basis, 'rounds': rounds, 'randomized': randomized,
               'decoded': benchmark(model, bits), 'raw': benchmark(model, bits, 'none')}
        summary.append(row)
        print(basis, rounds, randomized, row['decoded']['failures'], '/', len(bits),
              'detector fraction', round(row['decoded']['detector_event_fraction'], 5), flush=True)
    metrics = job.metrics() or {}
    run.update(status=job.status(), summary=summary,
               job_metrics={k:metrics[k] for k in ('timestamps', 'usage') if k in metrics})
    write_json(path, manifest)


def main():
    """Dispatch hardware phases; extra phases append jobs and raw shots to existing result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'reuse', 'submit', 'status', 'collect',
                                         'submit-extra', 'collect-extra', 'prepare-memory-pilot',
                                         'submit-memory-pilot', 'collect-memory-pilot', 'submit-ising', 'collect-ising',
                                         'prepare-ancilla', 'submit-ancilla', 'collect-ancilla'])
    parser.add_argument('--out', default='runs/ibm-surface-d7-10shots')
    parser.add_argument('--circuit', default='surface-d7-z.stim')
    parser.add_argument('--account-run', default='runs/conda-2026-11-04/ibm-d3')
    parser.add_argument('--reuse-from', default='runs/ibm-surface-d7-10shots')
    parser.add_argument('--shots', type=int, default=10, help='For prepare/reuse; submit uses the saved manifest')
    parser.add_argument('--total-shots', type=int, help='Total shots for both arms and all conditions at one distance')
    parser.add_argument('--distance', type=int, choices=(3, 5), default=3, help='Distance for the ancilla comparison')
    parser.add_argument('--experiment', default='shots1000', help='Key for an in-place added experiment')
    parser.add_argument('--comparison', default='runs/ibm-surface-d7-10-vs-100')
    parser.add_argument('--decoder-noise', type=float, help='Noise-model prior only; never sets native IBM noise')
    args = parser.parse_args()
    if not 1 <= args.shots <= 1000:
        parser.error('This trial helper accepts between 1 and 1000 shots')
    if args.decoder_noise is not None and not 0 <= args.decoder_noise < 1:
        parser.error('Decoder noise must be in [0, 1)')
    out = Path(args.out)
    # Reuse copies an already validated ISA circuit instead of rerunning routing.
    if args.phase == 'reuse':
        source = Path(args.reuse_from)
        manifest = json.loads((source/'manifest.json').read_text())
        if hashlib.sha256((source/'isa.qpy').read_bytes()).hexdigest() != manifest['isa_sha256']:
            raise ValueError('Source compiled circuit fingerprint mismatch')
        c = load(source/'circuit.stim')
        if hashlib.sha256(str(c).encode()).hexdigest() != manifest['circuit_sha256']:
            raise ValueError('Source Stim circuit fingerprint mismatch')
        manifest['reused_from'] = str(source.resolve())
        manifest['previous_job_id'] = manifest.pop('job_id', None)
        manifest['shots'] = args.shots
        manifest['limitations'] = [s for s in manifest['limitations'] if not s.startswith('10 shots')]
        manifest['limitations'].append('Small samples do not establish fault-tolerant performance or threshold behavior.')
        out.mkdir(parents=True, exist_ok=False)
        for name in ['circuit.stim', 'circuit.qpy', 'isa.qpy']:
            shutil.copyfile(source/name, out/name)
        write_json(out/'manifest.json', manifest)
        print(f"Reused identical compiled circuit on {manifest['backend']}; requested shots: {args.shots}")
        return
    # Preparation validates the actual routed circuit before a real submission.
    if args.phase == 'prepare':
        previous = json.loads((Path(args.account_run)/'manifest.json').read_text())
        svc = service(previous['instance'])
        instance = next(i for i in svc.instances() if i['crn'] == previous['instance'])
        if instance['plan'] != 'open':
            raise ValueError('This trial requires the verified free Open Plan instance')
        c = load(args.circuit)
        qc = to_qiskit(c)
        backend = svc.least_busy(operational=True, simulator=False,
                                dynamic_circuits=True, min_num_qubits=qc.num_qubits)
        print('Preparing for', backend.name, flush=True)
        isa = generate_preset_pass_manager(backend=backend, optimization_level=1,
                                          seed_transpiler=42).run(qc)
        invalid = []
        for instruction in isa.data:
            name = instruction.operation.name
            if name == 'barrier':
                continue
            qs = tuple(isa.find_bit(q).index for q in instruction.qubits)
            if not backend.target.instruction_supported(operation_name=name, qargs=qs):
                invalid.append((name, qs))
        if invalid:
            raise ValueError(f'Unsupported compiled operations: {invalid[:10]}')
        # Validate the actual routed ISA circuit under ideal dynamics, not just the source.
        sim = AerSimulator(method='stabilizer')
        ideal = transpile(isa, sim, optimization_level=0, seed_transpiler=42)
        r = sim.run(ideal, shots=10, memory=True, seed_simulator=42).result()
        m = measurement_array(r.get_memory(), c.num_measurements)
        dets, obs = c.compile_m2d_converter().convert(measurements=m, separate_observables=True)
        if dets.any() or obs.any():
            raise ValueError('Routed circuit failed ideal detector/observable validation')
        duration, duration_error = None, None
        try:
            duration = isa.estimate_duration(backend.target, unit='s')
        except Exception as exc:
            duration_error = str(exc)
        out.mkdir(parents=True, exist_ok=False)
        c.to_file(str(out/'circuit.stim'))
        for name, circuit in [('circuit.qpy', qc), ('isa.qpy', isa)]:
            with (out/name).open('wb') as f:
                qpy.dump(circuit,f)
        manifest = {'instance':previous['instance'],'verified_plan':instance['plan'],
            'backend':backend.name,'shots':args.shots,'circuit_sha256':hashlib.sha256(str(c).encode()).hexdigest(),
            'isa_sha256':hashlib.sha256((out/'isa.qpy').read_bytes()).hexdigest(),
            'versions':versions(),'num_circuit_qubits':qc.num_qubits,
            'num_backend_qubits':backend.num_qubits,'isa_depth':isa.depth(),
            'isa_operations':dict(isa.count_ops()),'num_measurements_per_shot':c.num_measurements,
            'num_detectors_per_shot':c.num_detectors,'stim_to_qiskit_qubits':qc.metadata['stim_to_qiskit_qubits'],
            'layout':str(isa.layout),'estimated_static_duration_seconds':duration,
            'duration_estimate_limitation':duration_error,
            'ideal_routed_validation':{'shots':10,'detector_events':int(dets.sum()),
                                       'logical_flips':int(obs.sum())},
            'sampler_options':{'twirling':{'enable_gates':False,'enable_measure':False},
                'execution':{'init_qubits':True},'max_execution_time':60},
            'limitations':['Generic hardware routing; no claim of preserved circuit-level distance.',
                'Offline decoding only; no decoder-directed data-qubit recovery.',
                'Small samples do not establish fault-tolerant performance or threshold behavior.']}
        write_json(out/'manifest.json',manifest)
        print(json.dumps({k:v for k,v in manifest.items() if k not in
                          {'instance','layout','stim_to_qiskit_qubits'}},indent=2),flush=True)
        return
    manifest = json.loads((out/'manifest.json').read_text())
    if args.phase.endswith('-ancilla'):
        if args.phase != 'collect-ancilla':
            if args.total_shots is not None and (args.total_shots % 4 or not 96 <= args.total_shots <= 600000):
                parser.error('Ancilla total shots must be a multiple of four between 96 and 600,000')
            if args.total_shots is None and args.shots % 4:
                parser.error('Ancilla reset comparison needs a shot count divisible by four')
        ancilla_trial(args, out, manifest, service(manifest['instance'], prefer_saved=True))
        return
    if args.phase.endswith('-ising'):
        if args.phase == 'submit-ising' and args.shots < 2:
            parser.error('Ising needs at least two shots to estimate sampling uncertainty')
        ising_trial(args, out, manifest, service(manifest['instance'], prefer_saved=True))
        return
    if args.phase.endswith('-memory-pilot'):
        memory_pilot(args, out, manifest, service(manifest['instance'], prefer_saved=True))
        return
    svc = service(manifest['instance'])
    if args.phase in {'submit-extra', 'collect-extra'}:
        extra_experiment(args, out, manifest, svc)
        return
    # Legacy collection retains its original directory-based artifact format.
    if args.phase == 'collect':
        job = svc.job(manifest['job_id'])
        result = job.result()[0]
        c = load(out/'circuit.stim')
        if hashlib.sha256(str(c).encode()).hexdigest() != manifest['circuit_sha256']:
            raise ValueError('Source circuit changed since preflight')
        measurements = measurement_array(result.data.meas.get_bitstrings(), c.num_measurements)
        expected = manifest['shots']
        if len(measurements) != expected:
            raise ValueError(f'Expected exactly {expected} shots, received {len(measurements)}')
        np.save(out/'measurements.npy', measurements)
        reports = []
        for decoder in ['none', 'pymatching', 'pymatching-correlated']:
            report = benchmark(c, measurements, decoder)
            report.update(source='ibm', job_id=manifest['job_id'], backend=manifest['backend'],
                          metadata=result.metadata, versions=versions())
            write_json(out/f'benchmark-{decoder}.json', report)
            reports.append(report)
        write_json(out/'status.json',{'status':job.status(),'metrics':job.metrics()})
        lines = [f'# {expected}-shot distance-7 hardware trial', '',
            f"Backend `{manifest['backend']}`; job `{manifest['job_id']}`. Exactly {expected} hardware shots retrieved.", '',
            'Source patch: 49 data qubits, 48 ancillas, seven syndrome rounds; logical-Z memory.',
            f"Routed circuit: depth {manifest['isa_depth']}; operations {manifest['isa_operations']}.", '',
            '| Decoder | Logical failures | Error per shot | 95% Wilson interval |',
            '|---|---:|---:|---|']
        for r in reports:
            lo,hi=r['wilson_95_interval']
            lines.append(f"| {r['decoder']} | {r['failures']}/{expected} | {100*r['logical_error_per_shot']:.1f}% | {100*lo:.1f}%–{100*hi:.1f}% |")
        lines += ['', *['- '+s for s in manifest['limitations']],
            '- Decoder weights come from the source Stim model; routing/idle/readout errors are not fitted to the hardware.',
            '- Timings in benchmark JSON files are local CPU decode timings, not QPU or feedback latency.',
            '- All three decoders use the same hardware shot records; no extra QPU executions are needed for these comparisons.']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps(reports,indent=2,default=str),flush=True)
        return
    if args.phase == 'status':
        job = svc.job(manifest['job_id'])
        status = {'status':job.status(),'metrics':job.metrics()}
        write_json(out/'status.json',status)
        print(json.dumps(status,indent=2,default=str))
        return
    # Do not accidentally submit an already recorded trial a second time.
    if 'job_id' in manifest:
        raise ValueError('This trial is already submitted; refusing duplicate submission')
    if not 1 <= manifest['shots'] <= 1000 or manifest['verified_plan'] != 'open':
        raise ValueError('Invalid trial bounds')
    if hashlib.sha256((out/'isa.qpy').read_bytes()).hexdigest() != manifest['isa_sha256']:
        raise ValueError('Compiled circuit changed after preflight')
    with (out/'isa.qpy').open('rb') as f:
        isa, = qpy.load(f)
    backend = svc.backend(manifest['backend'])
    if not backend.status().operational:
        raise ValueError('Selected backend is no longer operational')
    current_instance = next(i for i in svc.instances() if i['crn'] == manifest['instance'])
    if current_instance['plan'] != 'open':
        raise ValueError('Submission requires the free Open Plan')
    # Explicit options keep twirling disabled; the manifest fixes the requested shot count.
    sampler = SamplerV2(mode=backend, options=manifest['sampler_options'])
    job = sampler.run([isa], shots=manifest['shots'])
    manifest['job_id'] = job.job_id()
    write_json(out/'manifest.json',manifest)
    (out/'report.md').write_text(
        f"# {manifest['shots']}-shot distance-7 hardware trial\n\n"
        f"Submitted exactly {manifest['shots']} shots to `{manifest['backend']}`. Job `{job.job_id()}`. Results pending.\n\n"
        f"Circuit depth: {manifest['isa_depth']}; two-qubit CZ gates: {manifest['isa_operations'].get('cz',0)}.\n\n"
        'Seven syndrome rounds; 97 source-circuit qubits; 385 recorded measurements per shot.\n\n'
        'Offline decoding only. Generic routing does not establish preserved circuit-level distance.\n',encoding='utf-8')
    print(f"Submitted exactly {manifest['shots']} shots:",job.job_id(),flush=True)


if __name__ == '__main__':
    main()
