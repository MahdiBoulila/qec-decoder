"""Offline decoder comparison using a frozen IBM measurement file. No network calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pymatching
import stim

from compare_pymatching import wilson
from pipeline import versions, write_json


class ExactDemDecoder:
    """Sum all independent DEM fault combinations by syndrome/logical class.

    Exponential in detector + observable count; intended only for tiny experiments.
    Separators within an error instruction do not make its components independent.
    """
    def __init__(self, model, max_bits=20):
        """Convolve independent DEM faults over joint detector and logical states, preserving correlations."""
        self.num_detectors = model.num_detectors
        self.num_observables = model.num_observables
        bits = self.num_detectors + self.num_observables
        if bits > max_bits:
            raise ValueError(f'Exact DEM decoder limited to {max_bits} state bits; got {bits}')
        # Low bits encode detectors; higher bits encode logical-observable flips.
        indices = np.arange(1 << bits, dtype=np.int64)
        dist = np.zeros(len(indices), dtype=np.float64)
        dist[0] = 1
        for instruction in model.flattened():
            if instruction.type != 'error':
                continue
            probability, = instruction.args_copy()
            mask = 0
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    mask ^= 1 << target.val
                elif target.is_logical_observable_id():
                    mask ^= 1 << (self.num_detectors + target.val)
                elif not target.is_separator():
                    raise ValueError(f'Unsupported DEM target {target}')
            # XOR convolution sums both occurrence and absence of this fault.
            dist = (1-probability)*dist + probability*dist[indices ^ mask]
        # Rows are logical classes and columns are complete detector syndromes.
        self.joint = dist.reshape(1 << self.num_observables, 1 << self.num_detectors)
        self.syndrome_probability = self.joint.sum(axis=0)
        self.logical_class = self.joint.argmax(axis=0)

    def syndrome_indices(self, detectors):
        """Encode each binary detector row as a little-endian integer lookup index."""
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError('Detector input shape mismatch')
        return detectors.astype(np.int64) @ (1 << np.arange(self.num_detectors, dtype=np.int64))

    def decode_batch(self, detectors):
        """Choose the most probable logical class after summing all compatible fault histories."""
        indices = self.syndrome_indices(detectors)
        if np.any(self.syndrome_probability[indices] == 0):
            raise ValueError('A supplied syndrome has zero probability under the DEM')
        classes = self.logical_class[indices]
        return ((classes[:, None] >> np.arange(self.num_observables)) & 1).astype(np.uint8)

    def logical_class_posteriors(self, detectors):
        """Return normalized logical-class probabilities conditional on each observed syndrome."""
        indices = self.syndrome_indices(detectors)
        return (self.joint[:, indices] / self.syndrome_probability[indices]).T


def main():
    """Replay saved hardware shots through fixed decoders without sending another QPU job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/conda-2026-11-04/ibm-d3')
    parser.add_argument('--out', default='runs/ibm-decoder-comparison')
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.out)
    manifest = json.loads((run/'manifest.json').read_text())
    circuit = stim.Circuit.from_file(str(run/'circuit.stim'))
    fingerprint = hashlib.sha256(str(circuit).encode()).hexdigest()
    if fingerprint != manifest['circuit_sha256']:
        raise ValueError('Circuit fingerprint differs from submission')
    measurements_file = run/'measurements.npy'
    measurements = np.load(measurements_file, allow_pickle=False)
    if measurements.ndim != 2 or measurements.shape[1] != circuit.num_measurements:
        raise ValueError('Invalid measurement shape')
    detectors, observed = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    # Un-decomposed model for exact summation; decomposition preserves correlations
    # for PyMatching's correlated option, but ordinary matching ignores them.
    exact_model = circuit.detector_error_model()
    matching_model = circuit.detector_error_model(decompose_errors=True)
    decoders, build_seconds = {}, {}
    # Use the same detector matrix for both matching variants.
    for name, correlated in [('ordinary',False), ('correlated',True)]:
        start = time.perf_counter()
        matching = pymatching.Matching.from_detector_error_model(
            matching_model, enable_correlations=correlated)
        matching.ensure_num_fault_ids(circuit.num_observables)
        build_seconds[name] = time.perf_counter()-start
        decoders[name] = lambda data, m=matching, corr=correlated: m.decode_batch(data, enable_correlations=corr)
    start = time.perf_counter()
    exact = ExactDemDecoder(exact_model)
    build_seconds['exact_dem_ml'] = time.perf_counter()-start
    decoders['exact_dem_ml'] = exact.decode_batch
    decoders['no_correction'] = lambda data: np.zeros((len(data), circuit.num_observables), dtype=np.uint8)
    build_seconds['no_correction'] = 0.0
    predictions, timings = {}, {name: [] for name in decoders}
    for name, decode in decoders.items():
        predictions[name] = decode(detectors)
    rng = np.random.default_rng(42)
    names = list(decoders)
    for _ in range(9):
        for index in rng.permutation(len(names)):
            name = names[index]
            start = time.perf_counter()
            prediction = decoders[name](detectors)
            timings[name].append(time.perf_counter()-start)
            np.testing.assert_array_equal(prediction, predictions[name])
    stats = []
    for name in names:
        failures = int(np.any(predictions[name] != observed, axis=1).sum())
        stats.append({'decoder':name,'shots':len(observed),'failures':failures,
            'logical_error_per_shot':failures/len(observed),
            'wilson_95_interval':wilson(failures,len(observed)),
            'predicted_logical_flips':int(predictions[name].sum()),
            'build_seconds':build_seconds[name],
            'batch_decode_seconds_median':float(np.median(timings[name])),
            'batch_decode_seconds_samples':timings[name]})
    # Compare per-shot predictions, not just totals of observed logical failures.
    disagreements = {f'{a} vs {b}':int(np.any(predictions[a]!=predictions[b],axis=1).sum())
        for i,a in enumerate(names) for b in names[i+1:]}
    result = {'source':'ibm','job_id':manifest['job_id'],'backend':manifest['backend'],
        'circuit_sha256':fingerprint,
        'measurements_sha256':hashlib.sha256(measurements_file.read_bytes()).hexdigest(),
        'measurements_source':str(measurements_file.resolve()),
        'num_detectors':circuit.num_detectors,'num_observables':circuit.num_observables,
        'raw_logical_errors':int(np.any(observed,axis=1).sum()),
        'detector_events':int(detectors.sum()),'versions':versions(),
        'results':stats,'pairwise_prediction_disagreements':disagreements,
        'notes':['All methods receive the same detector rows; logical outcomes are used only for scoring.',
            'Exact ML is optimal under the supplied independent-fault DEM, not necessarily under IBM hardware noise.',
            'The DEM uses the original assumed Stim noise, without fitting to these hardware labels.',
            'Graph/lookup construction and DEM extraction are excluded from batch lookup/decode timing.',
            'These are warmed host batch timings, not per-round or IBM feedback latency.',
            'No new QPU jobs or network requests were made.']}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'comparison.json',result)
    exact_model.to_file(str(out/'model.dem'))
    np.savez_compressed(out/'predictions.npz',detectors=detectors,observed=observed,
        exact_posteriors=exact.logical_class_posteriors(detectors),**predictions)
    with (out/'per-shot.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['shot_index','detectors','observed_logical_flips',*names])
        for index in range(len(observed)):
            encode = lambda row: ''.join(map(str,row.astype(int)))
            writer.writerow([index,encode(detectors[index]),encode(observed[index]),
                *[encode(predictions[name][index]) for name in names]])
    lines = ['# Decoders on the same IBM measurements','',
        f"IBM backend: `{manifest['backend']}`; job `{manifest['job_id']}`; {len(observed)} shots. This is the legacy five-qubit, three-round repetition experiment, not distance-7 surface-code hardware.",'',
        '| Decoder | Logical failures | Predicted flips | 95% Wilson interval | Warm batch median (µs) | Build (ms) |',
        '|---|---:|---:|---|---:|---:|']
    for s in stats:
        lo,hi=s['wilson_95_interval']
        lines.append(f"| {s['decoder']} | {s['failures']}/{s['shots']} | {s['predicted_logical_flips']} | {100*lo:.2f}%–{100*hi:.2f}% | {s['batch_decode_seconds_median']*1e6:.1f} | {s['build_seconds']*1e3:.2f} |")
    lines += ['',f"Pairwise prediction disagreements: `{disagreements}`.",'',
        'The raw logical observable never flipped in this hardware sample. Matching an always-no-flip baseline does not establish a decoder advantage; these shots cannot rank decoder accuracy reliably. The exact decoder sums probabilities over all DEM fault combinations in each logical class, instead of selecting a single minimum-weight explanation. With eight detector bits and one observable there are only 512 joint states, making exact summation practical. This algorithm is exponential and is deliberately limited to 20 total bits; it cannot be used this way for the distance-7 surface-code dataset.','',
        *['- '+note for note in result['notes']], '',
        'Files: `per-shot.csv` provides a row-by-row comparison; `predictions.npz` includes posterior logical-class probabilities; `comparison.json` records provenance and timings.','',
        'DEM semantics: https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
