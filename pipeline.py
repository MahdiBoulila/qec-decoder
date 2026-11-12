"""Crumble/Stim -> Qiskit -> measurements -> detectors -> decoder -> benchmarks."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import pymatching
import stim


def generated_memory(code='surface', distance=7, rounds=7, noise=0.01, basis='z'):
    """Build a repetition or rotated surface-code memory circuit with explicit noise locations."""
    if code not in {'surface', 'repetition'} or basis not in {'x', 'z'}:
        raise ValueError('Unknown code or basis')
    if distance < 3 or rounds < 1 or not 0 <= noise < 1:
        raise ValueError('Use distance >= 3, rounds >= 1, and 0 <= noise < 1')
    if code == 'surface' and distance % 2 == 0:
        raise ValueError('Use an odd distance for the rotated surface code')
    if code == 'repetition' and basis != 'z':
        raise ValueError('The repetition example supports only Z-basis memory')
    task = f'surface_code:rotated_memory_{basis}' if code == 'surface' else 'repetition_code:memory'
    return stim.Circuit.generated(task, distance=distance, rounds=rounds,
        after_clifford_depolarization=noise, before_round_data_depolarization=noise,
        before_measure_flip_probability=noise, after_reset_flip_probability=noise)


def qubit_mapping(circuit):
    # Stim lattice coordinates can use sparse integer labels; allocate only those labels.
    """Map sparse Stim labels to contiguous Qiskit wires without changing measurement order."""
    ids = sorted({t.value for op in circuit.without_noise().flattened()
                  for t in op.targets_copy() if t.is_qubit_target})
    return {q: index for index, q in enumerate(ids)}


def with_noise_probability(circuit, probability):
    """Replace generated scalar noise priors, preserving the ideal gate sequence."""
    if not 0 <= probability < 1:
        raise ValueError('Noise probability must be in [0, 1)')
    result = stim.Circuit()
    scalar_noise = {'X_ERROR', 'Y_ERROR', 'Z_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2'}
    for op in circuit.flattened():
        args = [probability] if op.name in scalar_noise else op.gate_args_copy()
        result.append(op.name, op.targets_copy(), args)
    if result.without_noise() != circuit.without_noise().flattened():
        raise ValueError('Noise change unexpectedly altered the ideal circuit')
    return result


def load(path):
    """Read a circuit and reject missing observables, missing detectors, or invalid detector parities."""
    circuit = stim.Circuit.from_file(str(path))
    if not circuit.num_detectors or not circuit.num_observables:
        raise ValueError('Circuit must contain DETECTOR and OBSERVABLE_INCLUDE annotations.')
    circuit.detector_error_model(decompose_errors=True)
    return circuit


def to_qiskit(circuit):
    """Preserve every measurement in its own classical bit, in Stim record order.

    Noise belongs to the decoder model and is removed for hardware execution.
    Unsupported gates fail closed; this is deliberately not a general converter.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    mapping = qubit_mapping(circuit)
    qc = QuantumCircuit(QuantumRegister(len(mapping), 'q'),
                        ClassicalRegister(circuit.num_measurements, 'meas'))
    qc.metadata = {'stim_to_qiskit_qubits': mapping}
    measurement = 0
    # These annotations define classical analysis and geometry, not hardware gates.
    annotations = {'QUBIT_COORDS', 'SHIFT_COORDS', 'DETECTOR', 'OBSERVABLE_INCLUDE'}
    single = {'H': 'h', 'X': 'x', 'Y': 'y', 'Z': 'z', 'S': 's', 'S_DAG': 'sdg'}
    pairs = {'CX': 'cx', 'CY': 'cy', 'CZ': 'cz', 'SWAP': 'swap'}
    for op in circuit.without_noise().flattened():
        name, targets = op.name, op.targets_copy()
        if name in annotations:
            continue
        # Preserve circuit layer boundaries; a barrier is not a duration estimate.
        if name == 'TICK':
            qc.barrier()
            continue
        if any(not t.is_qubit_target or t.is_inverted_result_target for t in targets):
            raise ValueError(f'Unsupported target (feedback, Pauli product, or inversion): {op}')
        qs = [mapping[t.value] for t in targets]
        if name in single:
            for q in qs:
                getattr(qc, single[name])(q)
        elif name in pairs:
            for a, b in zip(qs[::2], qs[1::2]):
                getattr(qc, pairs[name])(a, b)
        # Reset to zero, then rotate into the requested preparation basis.
        elif name in {'R', 'RX', 'RY'}:
            for q in qs:
                qc.reset(q)
                if name == 'RX':
                    qc.h(q)
                elif name == 'RY':
                    qc.h(q)
                    qc.s(q)
        # Rotate X/Y readout into Z; allocate a fresh bit for every measurement.
        elif name in {'M', 'MX', 'MY', 'MR', 'MRX', 'MRY'}:
            axis = name[-1] if name[-1] in 'XY' else 'Z'
            for q in qs:
                if axis == 'Y':
                    qc.sdg(q)
                if axis != 'Z':
                    qc.h(q)
                qc.measure(q, measurement)
                measurement += 1
                if name.startswith('MR'):
                    qc.reset(q)
                # Restore the measured eigenstate, or prepare +X/+Y after reset.
                if axis != 'Z':
                    qc.h(q)
                if axis == 'Y':
                    qc.s(q)
        else:
            raise ValueError(f'Unsupported Stim instruction: {op}')
    if measurement != circuit.num_measurements:
        raise ValueError('Measurement count mismatch')
    return qc


def measurement_array(bitstrings, width):
    # Qiskit prints highest classical bit first; Stim uses chronological order.
    """Convert per-shot Qiskit bitstrings into chronological Stim measurement columns."""
    rows = list(bitstrings)
    if not rows or any(len(s) != width or set(s) - {'0', '1'} for s in rows):
        raise ValueError('Invalid per-shot measurement strings')
    return np.array([[int(b) for b in s[::-1]] for s in rows], dtype=np.bool_)


def benchmark(circuit, measurements, decoder='pymatching'):
    """Score logical predictions on fixed shot records; report CPU timing and binomial uncertainty."""
    if decoder not in {'pymatching', 'pymatching-correlated', 'none'}:
        raise ValueError('Unknown decoder')
    if measurements.ndim != 2 or measurements.shape[1] != circuit.num_measurements:
        raise ValueError('Measurement shape does not match Stim circuit')
    # Detection events are decoder inputs; observable flips are held-out scoring labels.
    dets, obs = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    start = time.perf_counter()
    # Correlated matching must be enabled at both graph construction and decoding.
    correlated = decoder == 'pymatching-correlated'
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True),
        enable_correlations=correlated) if decoder != 'none' else None
    if matching is not None:
        matching.ensure_num_fault_ids(circuit.num_observables)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = matching.decode_batch(dets, enable_correlations=correlated) if matching is not None else np.zeros_like(obs)
    elapsed = time.perf_counter() - start
    failures = int(np.count_nonzero(np.any(prediction != obs, axis=1)))
    n = len(obs)
    rate = failures / n
    # A zero observed failure rate still has a nonzero Wilson upper bound.
    z = 1.959963984540054
    center = (rate + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/(1+z*z/n)
    return {
        'decoder': decoder, 'shots': n, 'failures': failures,
        'logical_error_per_shot': rate, 'wilson_95_interval': [center-half, center+half],
        'raw_observable_error_per_shot': float(np.mean(np.any(obs, axis=1))),
        'detector_event_fraction': float(dets.mean()),
        'decoder_build_seconds': build_seconds, 'decode_seconds': elapsed,
        'decode_shots_per_second': n / elapsed if elapsed else None,
        'num_qubits': len(qubit_mapping(circuit)), 'stim_index_span': circuit.num_qubits,
        'num_detectors': circuit.num_detectors,
        'num_measurements': circuit.num_measurements,
    }


def write_json(path, obj):
    """Write a readable JSON snapshot, including provider metadata that may contain custom objects."""
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')


def versions():
    """Record package versions so hardware and simulation results can be reproduced."""
    return {p: importlib.metadata.version(p) for p in
            ['stim', 'pymatching', 'numpy', 'qiskit', 'qiskit-aer', 'qiskit-ibm-runtime']}


def service(instance, *, prefer_saved=False):
    """Connect using the chosen instance and local credentials without writing tokens to artifacts."""
    import os
    from qiskit_ibm_runtime import QiskitRuntimeService
    if prefer_saved:
        # Select the saved account by name so Qiskit cannot resolve a stale shell token instead.
        saved = QiskitRuntimeService.saved_accounts(default=True, channel='ibm_quantum_platform')
        if saved:
            name = next(iter(saved))
            return QiskitRuntimeService(name=name, instance=instance)
    # Require an explicit Open Plan instance; never silently pick a paid instance.
    token = os.environ.get('QISKIT_IBM_TOKEN')
    return QiskitRuntimeService(channel='ibm_quantum_platform', instance=instance,
                               **({'token': token} if token else {}))


def main():
    """Dispatch circuit generation, local simulation, preparation, submission, or result collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('example')
    gen.add_argument('--out', default='surface-d7-z.stim')
    gen.add_argument('--code', choices=['surface', 'repetition'], default='surface')
    gen.add_argument('--basis', choices=['x', 'z'], default='z')
    gen.add_argument('--distance', type=int, default=7)
    gen.add_argument('--rounds', type=int, default=7)
    gen.add_argument('--noise', type=float, default=0.01)
    for name in ['local', 'prepare', 'submit']:
        p = sub.add_parser(name)
        p.add_argument('circuit')
        p.add_argument('--out', required=True)
        if name != 'prepare':
            p.add_argument('--shots', type=int, default=256 if name == 'submit' else 10000)
        if name == 'local':
            p.add_argument('--engine', choices=['stim', 'aer'], default='stim')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--decoder', choices=['pymatching', 'pymatching-correlated', 'none'], default='pymatching')
        if name == 'submit':
            p.add_argument('--instance', required=True, help='Your free Open Plan instance CRN')
            p.add_argument('--backend', required=True)
    collect = sub.add_parser('collect')
    collect.add_argument('run_dir')
    collect.add_argument('--decoder', choices=['pymatching', 'pymatching-correlated', 'none'], default='pymatching')
    args = parser.parse_args()
    if args.command == 'example':
        try:
            c = generated_memory(args.code, args.distance, args.rounds, args.noise, args.basis)
        except ValueError as exc:
            parser.error(str(exc))
        c.to_file(args.out)
        return
    # Retrieve an existing job; this branch never submits extra hardware executions.
    if args.command == 'collect':
        out = Path(args.run_dir)
        manifest = json.loads((out/'manifest.json').read_text())
        c = load(out/'circuit.stim')
        if hashlib.sha256(str(c).encode()).hexdigest() != manifest['circuit_sha256']:
            raise ValueError('Circuit changed since submission')
        job = service(manifest['instance']).job(manifest['job_id'])
        result = job.result()[0]
        m = measurement_array(result.data.meas.get_bitstrings(), c.num_measurements)
        np.save(out/'measurements.npy', m)
        report = benchmark(c, m, args.decoder)
        report.update(source='ibm', job_id=manifest['job_id'], metadata=result.metadata,
                      versions=versions())
        write_json(out/f'benchmark-{args.decoder}.json', report)
        print(json.dumps(report, indent=2, default=str))
        return
    if getattr(args, 'shots', 1) <= 0:
        parser.error('shots must be positive')
    c = load(args.circuit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    c.to_file(str(out/'circuit.stim'))
    manifest = {'circuit_sha256': hashlib.sha256(str(c).encode()).hexdigest(),
                'versions': versions(), 'command': vars(args)}
    # Stim samples the specified noise; Aer checks the ideal translated circuit.
    if args.command == 'local':
        if args.engine == 'stim':
            m = c.compile_sampler(seed=args.seed).sample(args.shots)
        else:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            sim = AerSimulator(method='stabilizer')
            qc = transpile(to_qiskit(c), sim, seed_transpiler=args.seed)
            result = sim.run(qc, shots=args.shots, memory=True,
                             seed_simulator=args.seed).result()
            m = measurement_array(result.get_memory(), c.num_measurements)
        np.save(out/'measurements.npy', m)
        report = benchmark(c, m, args.decoder)
        report.update(source=args.engine, noise='Stim model' if args.engine == 'stim' else 'ideal',
                      seed=args.seed, versions=versions())
        write_json(out/'benchmark.json', report)
        write_json(out/'manifest.json', manifest)
        print(json.dumps(report, indent=2))
        return
    qc = to_qiskit(c)
    manifest['stim_to_qiskit_qubits'] = qc.metadata['stim_to_qiskit_qubits']
    from qiskit import qpy
    with (out/'circuit.qpy').open('wb') as f:
        qpy.dump(qc, f)
    (out/'circuit.txt').write_text(str(qc.draw(output='text')), encoding='utf-8')
    # Only this branch sends a new QPU workload and consumes account capacity.
    if args.command == 'submit':
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2
        svc = service(args.instance)
        backend = svc.backend(args.backend)
        isa = generate_preset_pass_manager(backend=backend, optimization_level=1,
                                          seed_transpiler=42).run(qc)
        with (out/'isa.qpy').open('wb') as f:
            qpy.dump(isa, f)
        manifest.update(instance=args.instance, backend=backend.name,
                        isa_depth=isa.depth(), isa_operations=dict(isa.count_ops()),
                        layout=str(isa.layout))
        write_json(out/'manifest.json', manifest)
        job = SamplerV2(mode=backend).run([isa], shots=args.shots)
        manifest['job_id'] = job.job_id()
        print('Submitted job:', job.job_id())
    write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
