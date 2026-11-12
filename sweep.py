"""Local surface-code memory benchmarks. Never submits IBM jobs."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import stim
from pipeline import benchmark, generated_memory, versions, write_json


def pauli_plus_records(stream, shots):
    """Read complete TFRecord frames and verify the original length and payload CRC32C."""
    import struct

    # TFRecord uses masked Castagnoli CRC32C, not the CRC32 used in ZIP files.
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)

    def masked_crc(data):
        """Compute TensorFlow's stored checksum for a byte sequence."""
        crc = 0xffffffff
        for byte in data:
            crc = table[(crc ^ byte) & 255] ^ (crc >> 8)
        crc ^= 0xffffffff
        return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff

    for _ in range(shots):
        header = stream.read(12)
        length, checksum = struct.unpack('<QI', header)
        if not 0 < length <= 1024**2 or masked_crc(header[:8]) != checksum:
            raise ValueError('Invalid Pauli+ TFRecord header')
        payload = stream.read(length)
        footer = stream.read(4)
        if len(payload) != length or masked_crc(payload) != struct.unpack('<I', footer)[0]:
            raise ValueError('Invalid Pauli+ TFRecord payload')
        yield header, payload, footer


def _pauli_plus_fields(data):
    """Read the protobuf wire types used by the published Example and TensorProto fields."""
    def varint(index):
        """Read a bounded protobuf integer and return the next offset."""
        value = 0
        for shift in range(0, 70, 7):
            byte = data[index]
            index += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, index
        raise ValueError('Invalid protobuf integer')

    fields = {}
    index = 0
    while index < len(data):
        tag, index = varint(index)
        wire = tag & 7
        if wire == 0:
            value, index = varint(index)
        elif wire == 2:
            length, index = varint(index)
            value = data[index:index+length]
            index += length
            if index > len(data):
                raise ValueError('Truncated protobuf field')
        else:
            raise ValueError(f'Unexpected Pauli+ protobuf wire type {wire}')
        fields.setdefault(tag >> 3, []).append(value)
    return fields


def load_pauli_plus_subset(path):
    """Load original d7 tensors from the saved literature JSON with NumPy and the standard library."""
    import base64
    import gzip
    import io

    registry = json.loads(Path(path).read_text(encoding='utf-8'))['literature_database']
    subset = registry['datasets']['gdm-qec-release']['d7_original_subset']
    packed = base64.b64decode(subset['gzip_tfrecord_base64'], validate=True)
    if hashlib.sha256(packed).hexdigest() != subset['sha256']:
        raise ValueError('Saved Pauli+ subset checksum mismatch')
    fields = _pauli_plus_fields
    rows = {}
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
        for _, payload, _ in pauli_plus_records(stream, subset['shots']):
            # Walk Example -> Features map -> bytes_list -> serialized TensorProto.
            for blob in fields(fields(payload)[1][0])[1]:
                entry = fields(blob)
                name = entry[1][0].decode()
                tensor = fields(fields(fields(entry[2][0])[1][0])[1][0])
                shape = [fields(dim)[1][0] for dim in fields(tensor[2][0]).get(2, [])]
                dtype = {1: '<f4', 4: 'u1'}[tensor[1][0]]
                rows.setdefault(name, []).append(np.frombuffer(tensor[4][0], dtype=dtype).reshape(shape))
        if stream.read(1):
            raise ValueError('Unexpected trailing Pauli+ records')
    return {name: np.stack(values) for name, values in rows.items()}


def si1000_geometry(circuit, rounds):
    """Map actual measurement records and boundary detectors to round/stabilizer coordinates."""
    by_qubit, measured_qubits, detector_records = {}, [], []
    # Resolve DETECTOR rec offsets at their instruction location, before any later readout.
    for op in circuit.flattened():
        if op.name in {'M', 'MX', 'MY', 'MR', 'MRX', 'MRY'}:
            for target in op.targets_copy():
                qubit = target.value
                by_qubit.setdefault(qubit, []).append(len(measured_qubits))
                measured_qubits.append(qubit)
        elif op.name == 'DETECTOR':
            detector_records.append([len(measured_qubits)+t.value for t in op.targets_copy()])
    ancillas = sorted(q for q, indices in by_qubit.items() if len(indices) == rounds)
    data = {q for q, indices in by_qubit.items() if len(indices) == 1}
    if rounds < 2 or not ancillas or set(ancillas) & data:
        raise ValueError('Published SI1000 loader requires distinct repeatedly measured ancillas')
    qubit_coords = circuit.get_final_qubit_coordinates()
    coordinate_to_slot = {tuple(qubit_coords[q][:2]): i for i, q in enumerate(ancillas)}
    if len(coordinate_to_slot) != len(ancillas):
        raise ValueError('Ambiguous stabilizer coordinates')
    slots, occupied = [], set()
    for detector, coord in circuit.get_detector_coordinates().items():
        # Some published circuits give final boundary checks the same time coordinate as
        # the last ancilla checks. Data-record support identifies the boundary reliably.
        final = any(measured_qubits[i] in data for i in detector_records[detector])
        frame = rounds if final else int(coord[2])
        slot = coordinate_to_slot[tuple(coord[:2])]
        if not 0 <= frame <= rounds or (frame, slot) in occupied:
            raise ValueError('Detector-to-frame mapping is ambiguous')
        slots.append([frame, slot])
        occupied.add((frame, slot))
    return {'ancilla_qubits': ancillas,
        'ancilla_measurement_indices': [[by_qubit[q][r] for q in ancillas] for r in range(rounds)],
        'stabilizer_coordinates': [qubit_coords[q][:2] for q in ancillas],
        'detector_slots': slots, 'detector_frame_count': rounds+1,
        'final_boundary_detector_count': sum(frame == rounds for frame, _ in slots)}


def si1000_training_arrays(case, split):
    """Return aligned neural inputs and separate labels from a saved, checksum-verified split."""
    from hardware_trial import decode_measurements
    saved = case['splits'][split]
    circuit = stim.Circuit(case['circuit'])
    packed = decode_measurements(saved['packed_measurements'])
    if packed.dtype != np.uint8 or packed.shape != (saved['shots'], (circuit.num_measurements+7)//8):
        raise ValueError('Packed SI1000 measurements have an invalid shape or dtype')
    measurements = np.unpackbits(packed, axis=1, bitorder='little', count=circuit.num_measurements).astype(bool)
    detectors, labels = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
    if hashlib.sha256(np.packbits(detectors, axis=1, bitorder='little').tobytes()).hexdigest() != saved['detectors_sha256']:
        raise ValueError('Regenerated SI1000 detector checksum mismatch')
    if hashlib.sha256(np.packbits(labels, axis=1, bitorder='little').tobytes()).hexdigest() != saved['labels_sha256']:
        raise ValueError('Regenerated SI1000 label checksum mismatch')
    geometry = case['geometry']
    slots = geometry['detector_slots']
    frames, stabilizers = geometry['detector_frame_count'], len(geometry['ancilla_qubits'])
    events = np.zeros((saved['shots'], frames, stabilizers), dtype=bool)
    mask = np.zeros((frames, stabilizers), dtype=bool)
    for index, (frame, slot) in enumerate(slots):
        events[:, frame, slot] = detectors[:, index]
        mask[frame, slot] = True
    # Missing initial/final checks are masked, not misrepresented as measured zeroes.
    # Logical-flip answer labels and full final data readout are not input features.
    return {'inputs': {'ancilla_measurements': measurements[:, geometry['ancilla_measurement_indices']],
        'detector_events': events, 'detector_mask': mask,
        'stabilizer_coordinates': np.asarray(geometry['stabilizer_coordinates'])}, 'labels': labels}


def create_si1000_dataset(document, key, distances, bases, noises, split_shots, seed):
    """Sample pinned published SI1000 circuits, with disjoint seeded splits and held-out baselines."""
    from hardware_trial import encode_measurements
    if key in document.get('local_datasets', {}):
        raise ValueError('Dataset key already exists; refusing to overwrite saved samples')
    if len(split_shots) != 3 or min(split_shots) < 1 or seed < 0:
        raise ValueError('Use three positive split sizes and a nonnegative seed')
    source = document['si1000_published_sources']
    selected = [row for row in source['circuits'] if row['distance'] in distances
        and row['basis'] in bases and row['noise'] in noises]
    if len(selected) != len(set(distances))*len(set(bases))*len(set(noises)):
        raise ValueError('Requested SI1000 configuration is not in the pinned published circuits')
    result = {'schema_version': 1, 'source': 'local Stim simulation', 'noise_model': 'SI1000',
        'parameters': {'distances': distances, 'bases': bases, 'noises': noises,
            'rounds': 'equal to distance in these unmodified published circuits',
            'split_shots': dict(zip(['train', 'validation', 'test'], split_shots)), 'master_seed': seed},
        'versions': versions(), 'source_circuits': {k: v for k, v in source.items() if k != 'circuits'},
        'limitations': ['Our independent samples, not AlphaQubit original samples or hardware data.',
            'Pauli-only SI1000; no analogue I/Q, leakage, crosstalk or fault-location truth.',
            'Different distances/bases need geometry-aware models; final boundary inputs are retained.',
            'Baselines are scored only on the test split and are per shot, not fitted per round.',
            'These circuits reset ancillas between rounds; no reset-method ablation is performed.'], 'cases': []}
    for row in selected:
        if hashlib.sha256(row['circuit'].encode()).hexdigest() != row['sha256']:
            raise ValueError('Published SI1000 source circuit checksum mismatch')
        circuit = stim.Circuit(row['circuit'])
        # Check deterministic boundary annotations before spending effort generating training data.
        ideal = circuit.without_noise()
        clean = ideal.compile_sampler(seed=seed).sample(64)
        clean_dets, clean_obs = ideal.compile_m2d_converter().convert(measurements=clean, separate_observables=True)
        if np.any(clean_dets) or np.any(clean_obs):
            raise ValueError('Ideal SI1000 source circuit fails detector or logical validation')
        case = dict(row)
        case.update(id=f"si1000-d{row['distance']}-r{row['rounds']}-{row['basis']}-p{row['noise']}",
            geometry=si1000_geometry(circuit, row['rounds']), splits={})
        for split_id, (split, shots) in enumerate(zip(['train', 'validation', 'test'], split_shots)):
            # Derive seeds from configuration, not iteration order, and use independent split streams.
            entropy = [seed, row['distance'], row['rounds'], int(row['basis'] == 'z'),
                round(row['noise']*1000000), split_id]
            stream_seed = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
            measurements = circuit.compile_sampler(seed=stream_seed).sample(shots)
            detectors, labels = circuit.compile_m2d_converter().convert(measurements=measurements, separate_observables=True)
            case['splits'][split] = {'shots': shots, 'seed': stream_seed,
                'packed_measurements': encode_measurements(np.packbits(measurements, axis=1, bitorder='little')),
                'measurement_shape': list(measurements.shape),
                'detectors_sha256': hashlib.sha256(np.packbits(detectors, axis=1, bitorder='little').tobytes()).hexdigest(),
                'labels_sha256': hashlib.sha256(np.packbits(labels, axis=1, bitorder='little').tobytes()).hexdigest()}
            if split == 'test':
                # All baseline methods see identical held-out test records and the same noise model.
                case['test_benchmarks'] = [benchmark(circuit, measurements, decoder)
                    for decoder in ['none', 'pymatching', 'pymatching-correlated']]
        result['cases'].append(case)
        print(case['id'], [(r['decoder'], r['failures']) for r in case['test_benchmarks']], flush=True)
    result['total_shots'] = sum(split_shots)*len(selected)
    document.setdefault('local_datasets', {})[key] = result
    return result


def main():
    """Sample each circuit once, reuse its records across decoders, and export local benchmark summaries."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/sweep')
    p.add_argument('--shots', type=int, default=10000)
    p.add_argument('--shot-counts', type=int, nargs='+', help='Independent shot counts for an appended sweep')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise', type=float, default=0.01)
    p.add_argument('--append-key', help='Append into an existing sweep JSON; create no files')
    p.add_argument('--code', choices=['surface', 'repetition'], default='surface')
    p.add_argument('--distances', type=int, nargs='+', default=[7])
    p.add_argument('--rounds', type=int, nargs='+', default=[3, 7, 11])
    p.add_argument('--bases', choices=['x', 'z'], nargs='+', default=['x', 'z'])
    p.add_argument('--si1000-dataset-key', help='Append a seeded SI1000 training dataset to an existing benchmark.json')
    p.add_argument('--si1000-noises', type=float, nargs='+', default=[0.001, 0.003], help='Published SI1000 points; separate from --noise')
    p.add_argument('--split-shots', type=int, nargs=3, default=[30000, 10000, 10000], metavar=('TRAIN', 'VALIDATION', 'TEST'))
    args = p.parse_args()
    if args.si1000_dataset_key:
        # This path never creates directories, downloads code, overwrites datasets or submits IBM jobs.
        if args.append_key or args.code != 'surface':
            p.error('SI1000 dataset mode requires surface code and no --append-key')
        path = Path(args.out)/'benchmark.json'
        if not path.is_file():
            p.error('SI1000 dataset mode requires an existing benchmark.json with pinned source circuits')
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
            result = create_si1000_dataset(document, args.si1000_dataset_key, args.distances,
                args.bases, args.si1000_noises, args.split_shots, args.seed)
        except (KeyError, ValueError) as exc:
            p.error(str(exc))
        write_json(path, document)
        print(f"Saved {result['total_shots']} independent SI1000 shots into {path}")
        return
    if (args.shots < 1 or not 0 <= args.noise < 1 or min(args.distances) < 3
            or min(args.rounds) < 1):
        p.error('Use positive shots/rounds, distances >= 3, and 0 <= noise < 1')
    if args.code == 'surface' and any(d % 2 == 0 for d in args.distances):
        p.error('Rotated surface-code distances must be odd')
    if args.code == 'repetition' and args.bases != ['z']:
        p.error('For repetition memory specify --bases z')
    out = Path(args.out)
    if args.shot_counts and (not args.append_key or min(args.shot_counts) < 1 or
                            len(set(args.shot_counts)) != len(args.shot_counts)):
        p.error('--shot-counts requires append mode and distinct positive counts')
    if args.append_key:
        # Freeze old results and store new raw shots inside the same JSON document.
        from hardware_trial import encode_measurements
        path = out/'benchmark.json'
        if not path.is_file():
            p.error('Append mode requires an existing benchmark.json')
        previous = json.loads(path.read_text())
        additional = previous.setdefault('additional_sweeps', {})
        if args.append_key in additional:
            p.error('Sweep key already exists; choose a new explicit experiment key')
        trial = {'parameters': vars(args), 'source': 'stim', 'versions': versions(), 'cases': []}
        for distance in args.distances:
            for rounds in args.rounds:
                for basis in args.bases:
                    c = generated_memory(args.code, distance, rounds, args.noise, basis)
                    for shots in args.shot_counts or [args.shots]:
                        # Give each shot count its own reproducible sample stream.
                        seed = args.seed
                        if args.shot_counts:
                            seed = int(np.random.SeedSequence([seed, distance, rounds, basis == 'z', shots])
                                       .generate_state(1, dtype=np.uint64)[0])
                        started_at = datetime.now(timezone.utc).isoformat()
                        m = c.compile_sampler(seed=seed).sample(shots)
                        decoders = ['none', 'pymatching'] if args.shot_counts else ['none', 'pymatching', 'pymatching-correlated']
                        results = [benchmark(c, m, decoder) for decoder in decoders]
                        identifier = f'{args.code}-d{distance}-r{rounds}-{basis}'
                        case = {'id': identifier, 'distance': distance, 'rounds': rounds, 'basis': basis,
                                'noise': args.noise, 'circuit': str(c), 'shots': shots, 'seed': seed,
                                'started_at': started_at, 'results': results}
                        if args.shot_counts:
                            case['id'] += f'-n{shots}'
                            # Pack each shot before compression to keep large histories replayable.
                            case['packed_measurements'] = encode_measurements(np.packbits(m, axis=1, bitorder='little'))
                        else:
                            case['measurements'] = encode_measurements(m)
                        trial['cases'].append(case)
                        print(case['id'], [(r['decoder'], r['failures']) for r in results], flush=True)
        additional[args.append_key] = trial
        write_json(path, previous)
        return
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    # Keep basis and storage duration explicit rather than pooling different experiments.
    for distance in args.distances:
        for rounds in args.rounds:
            for basis in args.bases:
                c = generated_memory(args.code, distance, rounds, args.noise, basis)
                stem = f'{args.code}-d{distance}-r{rounds}-{basis}'
                c.to_file(str(out/f'{stem}.stim'))
                m = c.compile_sampler(seed=args.seed).sample(args.shots)
                np.save(out/f'{stem}-measurements.npy', m)
                # Reuse the same measurements for a paired decoder comparison.
                for decoder in ['none', 'pymatching']:
                    row = benchmark(c, m, decoder)
                    row.update(code=args.code, basis=basis, distance=distance,
                               rounds=rounds, noise=args.noise)
                    rows.append(row)
    write_json(out/'benchmark.json', {'parameters': vars(args), 'versions': versions(),
                                     'source': 'stim', 'results': rows})
    with (out/'benchmark.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
