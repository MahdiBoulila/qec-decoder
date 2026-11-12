"""Compare PyMatching 2 options on saved, identical surface-code shot records."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import re
import sys
import time
from importlib.metadata import version
from zipfile import ZipFile

import numpy as np
import pymatching
from scipy.stats import binomtest
import stim

from pipeline import versions, write_json


def wilson(failures, shots):
    """Calculate a two-sided 95% Wilson interval for a binomial logical failure probability."""
    p = failures / shots
    z = 1.959963984540054
    den = 1 + z*z/shots
    center = (p + z*z/(2*shots))/den
    half = z*np.sqrt(p*(1-p)/shots + z*z/(4*shots*shots))/den
    return [float(center-half), float(center+half)]


def compare_sycamore(case, shot_limit=None, seed=42, repeats=3):
    """Benchmark local decoders against one archived circuit DEM and identical measured shots."""
    from beliefmatching import BeliefMatching
    from compare_hardware_shots import SYCAMORE_ARCHIVE, sycamore_labels
    from replay_ibm_decoders import ExactDemDecoder

    if not re.fullmatch(r'surface_code_b[XZ]_d[35]_r\d{2}_center_\d+_\d+', case):
        raise ValueError('Expected a Sycamore surface-code archive configuration ID')
    if repeats < 1 or (shot_limit is not None and shot_limit < 1):
        raise ValueError('Shot and repeat counts must be positive')
    with ZipFile(SYCAMORE_ARCHIVE) as archive:
        model_bytes = archive.read(f'{case}/circuit_detector_error_model.dem')
        model = stim.DetectorErrorModel(model_bytes.decode())
        observed = sycamore_labels(archive, case, 'obs_flips_actual.01')
        packed = np.frombuffer(archive.read(f'{case}/detection_events.b8'), dtype=np.uint8)
    if model.num_observables != 1:
        raise ValueError('Expected a single logical observable')
    packed = packed.reshape(len(observed), (model.num_detectors+7)//8)
    indices = np.arange(len(observed))
    if shot_limit is not None and shot_limit < len(observed):
        # Keep the fixed random sample in acquisition order and use it for every decoder.
        indices = np.sort(np.random.default_rng(seed).choice(indices, shot_limit, replace=False))
    detectors = np.unpackbits(packed[indices], count=model.num_detectors,
                              axis=1, bitorder='little').astype(bool)
    observed = observed[indices]

    def build_matching(correlated):
        """Configure the matching graph once using the shared model."""
        graph = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
        graph.ensure_num_fault_ids(model.num_observables)
        return lambda batch: graph.decode_batch(batch, enable_correlations=correlated)

    builders = {
        'PyMatching': lambda: build_matching(False),
        'Correlated PyMatching': lambda: build_matching(True),
        'Belief matching': lambda: BeliefMatching(model, max_bp_iters=20, bp_method='product_sum').decode_batch,
    }
    # The exact reference contracts the DEM's fault factors as a dense joint tensor.
    # A separator is a graph decomposition hint and does not split a fault in this contraction.
    exact_bits = model.num_detectors + model.num_observables
    if exact_bits <= 20:
        builders['Exact tensor contraction (lookup)'] = lambda: ExactDemDecoder(model).decode_batch
    results = {'Raw': {'failures': int(observed.sum()), 'rate': float(observed.mean()),
                       'interval': wilson(int(observed.sum()), len(observed))}}
    failure_masks = {}
    for name, build in builders.items():
        start = time.perf_counter()
        decode = build()
        setup = time.perf_counter() - start
        decode(detectors[:min(16, len(detectors))])
        timings = []
        predictions = None
        for _ in range(repeats):
            start = time.perf_counter()
            current = decode(detectors)
            timings.append(time.perf_counter() - start)
            if predictions is not None and not np.array_equal(predictions, current):
                raise ValueError(f'{name} predictions changed between timing repeats')
            predictions = current
        if predictions.shape != (len(observed), 1) or not np.isin(predictions, [0, 1]).all():
            raise ValueError(f'{name} returned invalid logical predictions')
        failures = predictions[:, 0].astype(bool) ^ observed
        failure_masks[name] = failures
        count = int(failures.sum())
        median_seconds = float(np.median(timings))
        results[name] = {'failures': count, 'rate': count/len(observed),
            'interval': wilson(count, len(observed)), 'setup_seconds': setup,
            'decode_seconds_samples': timings, 'decode_seconds_median': median_seconds,
            'batch_us_per_shot': 1e6*median_seconds/len(observed),
            'setup_plus_decode_us_per_shot': 1e6*(setup+median_seconds)/len(observed),
            'prediction_sha256': hashlib.sha256(np.packbits(predictions).tobytes()).hexdigest()}
        print(f'{case} | {name} | {count}/{len(observed)} | {results[name]["batch_us_per_shot"]:.2f} us/shot', flush=True)

    paired = {}
    baseline = failure_masks['PyMatching']
    for name, failures in failure_masks.items():
        if name == 'PyMatching':
            continue
        wins = int((baseline & ~failures).sum())
        losses = int((~baseline & failures).sum())
        paired[name] = {'only_this_decoder_correct': wins, 'only_pymatching_correct': losses,
                        'mcnemar_exact_p': binomtest(wins, wins+losses, 0.5).pvalue if wins+losses else 1.0}
    return {'configuration': case, 'shots': len(observed), 'source_shots': len(packed),
        'selected_shot_indices': indices.tolist() if len(indices) < len(packed) else 'all',
        'seed': seed, 'repeats': repeats, 'num_detectors': model.num_detectors,
        'dem_member': f'{case}/circuit_detector_error_model.dem',
        'dem_sha256': hashlib.sha256(model_bytes).hexdigest(), 'results': results, 'paired_vs_pymatching': paired,
        'versions': versions() | {'beliefmatching': version('beliefmatching'), 'ldpc': version('ldpc')},
        'platform': platform.platform(), 'processor': platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER'),
        'timing_scope': 'CPU batch decoding after warm-up, excluding ZIP IO and scoring. Setup timed separately.',
        'exact_contraction_skipped': f'{exact_bits} state bits exceed the dense reference limit of 20' if exact_bits > 20 else None}


def compare_case(circuit, measurements, repeats=5, latency_shots=1000, seed=42):
    """Compare four decoder configurations on identical shots, separating accuracy and host timing."""
    start = time.perf_counter()
    dets, obs = circuit.compile_m2d_converter().convert(
        measurements=measurements, separate_observables=True)
    conversion_seconds = time.perf_counter() - start
    start = time.perf_counter()
    # PyMatching uses little-endian bit packing; padding bits are not detectors.
    packed = np.packbits(dets, axis=1, bitorder='little')
    packing_seconds = time.perf_counter() - start
    start = time.perf_counter()
    dem = circuit.detector_error_model(decompose_errors=True)
    dem_seconds = time.perf_counter() - start
    # Construct each graph once; exclude construction from repeated decode timing.
    graphs, builds = {}, {}
    for correlated in [False, True]:
        start = time.perf_counter()
        matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlated)
        matching.ensure_num_fault_ids(circuit.num_observables)
        builds[correlated] = time.perf_counter() - start
        graphs[correlated] = matching
    configs = [(False, False), (False, True), (True, False), (True, True)]

    def decode(config, warmup=False):
        """Run the chosen graph with matching packed-input and packed-output flags."""
        correlated, bit_packed = config
        data = packed if bit_packed else dets
        if warmup:
            data = data[:min(128, len(data))]
        return graphs[correlated].decode_batch(data, bit_packed_shots=bit_packed,
            bit_packed_predictions=bit_packed, enable_correlations=correlated)

    for config in configs:
        decode(config, warmup=True)
    timings = {c: [] for c in configs}
    predictions = {}
    rng = np.random.default_rng(seed)
    # Interleave configurations to reduce systematic order/temperature bias.
    for repeat in range(repeats):
        for index in rng.permutation(len(configs)):
            config = configs[index]
            start = time.perf_counter()
            pred = decode(config)
            timings[config].append(time.perf_counter() - start)
            if repeat == 0:
                predictions[config] = (np.unpackbits(pred, axis=1, bitorder='little')
                    [:, :circuit.num_observables] if config[1] else pred)
            else:
                expanded = (np.unpackbits(pred, axis=1, bitorder='little')
                    [:, :circuit.num_observables] if config[1] else pred)
                np.testing.assert_array_equal(expanded, predictions[config])
    # Packing must change storage only, not predicted logical corrections.
    for correlated in [False, True]:
        np.testing.assert_array_equal(predictions[correlated, False],
                                      predictions[correlated, True])
    # Whole-shot Python decode latency: not streaming, not an IBM control-loop measurement.
    sample_indices = rng.choice(len(dets), min(latency_shots, len(dets)), replace=False)
    latencies = {False: [], True: []}
    for index in sample_indices:
        for correlated in rng.permutation([False, True]):
            start = time.perf_counter_ns()
            p = graphs[correlated].decode(dets[index], enable_correlations=bool(correlated))
            latencies[bool(correlated)].append((time.perf_counter_ns()-start)/1000)
            np.testing.assert_array_equal(p, predictions[bool(correlated), False][index])
    rows = []
    failed = {}
    for config in configs:
        correlated, bit_packed = config
        mask = np.any(predictions[config] != obs, axis=1)
        failed[config] = mask
        failures = int(mask.sum())
        median = float(np.median(timings[config]))
        rows.append({'algorithm': 'correlated' if correlated else 'ordinary',
            'bit_packed': bit_packed, 'shots': len(dets), 'failures': failures,
            'logical_error_per_shot': failures/len(dets),
            'wilson_95_interval': wilson(failures, len(dets)),
            'graph_build_seconds': builds[correlated],
            'batch_decode_seconds_samples': timings[config],
            'batch_decode_seconds_median': median,
            'batch_shots_per_second': len(dets)/median,
            'single_shot_decode_us': dict(zip(['p50', 'p95', 'p99'],
                [float(x) for x in np.percentile(latencies[correlated], [50,95,99])]))})
    ordinary, correlated = failed[False, False], failed[True, False]
    # Count paired wins and losses; marginal error rates alone hide disagreements.
    rescued = int(np.count_nonzero(ordinary & ~correlated))
    harmed = int(np.count_nonzero(~ordinary & correlated))
    discordant = rescued + harmed
    paired = {'ordinary_failed_correlated_succeeded': rescued,
              'ordinary_succeeded_correlated_failed': harmed,
              'net_failures_avoided': rescued-harmed,
              'mcnemar_exact_two_sided_p_unadjusted': float(binomtest(rescued, discordant, 0.5).pvalue)
                  if discordant else 1.0}
    return {'measurement_to_detector_seconds': conversion_seconds,
            'detector_packing_seconds': packing_seconds, 'dem_build_seconds': dem_seconds,
            'unpacked_detector_bytes': dets.nbytes, 'packed_detector_bytes': packed.nbytes,
            'single_shot_latency_sample_size': len(sample_indices),
            'variants': rows, 'paired_accuracy_comparison': paired}


def report_markdown(result):
    """Render performance results with explicit simulation provenance and timing limitations."""
    lines = ['# PyMatching 2 option comparison', '',
        'All variants use the same saved shot records per circuit. Source: local Stim simulations, distance 7, noise parameter 0.005. No IBM job was submitted.', '',
        '| Memory basis | Rounds | Algorithm | Packed | Failures / shots | Error rate | Batch kshots/s | Whole-shot p95 (µs) |',
        '|---|---:|---|---|---:|---:|---:|---:|']
    for case in result['cases']:
        for v in case['variants']:
            lines.append(f"| {case['basis'].upper()} | {case['rounds']} | {v['algorithm']} | {v['bit_packed']} | {v['failures']} / {v['shots']} | {100*v['logical_error_per_shot']:.2f}% | {v['batch_shots_per_second']/1000:.1f} | {v['single_shot_decode_us']['p95']:.1f} |")
    lines += ['', '## Interpretation', '',
        '- Packed and unpacked predictions were checked for exact equality within each algorithm. Packing changes representation, not accuracy.',
        '- Batch timings are medians of warmed, interleaved runs. Graph construction, measurement conversion and packing are measured separately; they are excluded from decode throughput.',
        '- Single-shot latencies include a Python decode call for an entire completed shot. They do not measure causal per-round decoding or IBM measurement/transport/feedforward latency. Packed rows share the same unpacked single-shot latency measurement for that algorithm.',
        '- Timing is host-dependent and can be affected by other processes. See JSON for all repeated measurements, graph build time, confidence intervals, and paired failure comparisons.',
        '- Paired p-values in JSON are exploratory and unadjusted across cases. Simulated noise is the same model used for decoder weights; hardware accuracy may differ.',
        '- Correlated matching uses the decomposed detector error model\'s preserved fault correlations. Both graph construction and decoding enable correlations.',
        '', '## Environment', '', f"Python: {result['python_executable']}", '']
    lines.extend(f'- {k}: {v}' for k,v in result['versions'].items())
    lines += ['', 'Sources: [PyMatching releases](https://github.com/oscarhiggott/PyMatching/releases), [PyMatching source and API](https://github.com/oscarhiggott/PyMatching/blob/master/src/pymatching/matching.py), [Sparse blossom](https://arxiv.org/abs/2303.15933).']
    return '\n'.join(lines)+'\n'


def validate_literature(database):
    """Check citation links, unique identifiers, and metric units before presenting literature."""
    # A null metric means unextracted or unreported; it must never become a zero error rate.
    for key in ('sources', 'decoders', 'benchmarks'):
        ids = [row['id'] for row in database[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate literature {key} identifier')
    sources = {row['id'] for row in database['sources']}
    decoders = {row['id'] for row in database['decoders']}
    for row in database['benchmarks']:
        if row['source_id'] not in sources or row['decoder_id'] not in decoders:
            raise ValueError('Unknown benchmark citation or decoder')
        if not row.get('source_locator') or not row.get('workload'):
            raise ValueError('Literature benchmarks need a source location and workload')
        rate = row['logical_error_per_round']
        if rate is not None and not 0 <= rate <= 1:
            raise ValueError('Logical error per round must be a probability')
        speed = row['speed']
        if speed is not None and (speed['unit'] != 'us' or speed['value'] <= 0):
            raise ValueError('Literature timing must be positive and expressed in microseconds')


def literature_markdown(database, decoder_filter=None):
    """Render scoped literature observations without ranking unlike experimental workloads."""
    validate_literature(database)
    decoders = {row['id']: row for row in database['decoders']}
    sources = {row['id']: row for row in database['sources']}
    lines = ['## Decoder literature database', '',
        f"Reviewed {database['reviewed_on']}; curated seed, not an exhaustive ranking.", '',
        'Rates are probabilities per round. Timing labels distinguish throughput from latency. '
        'An em dash means no numeric value was extracted; it does not mean zero.', '',
        '| Decoder | Evidence / workload | Logical error / round | Timing (microseconds) | Source |',
        '|---|---|---:|---|---|']
    matched = []
    for row in database['benchmarks']:
        decoder = decoders[row['decoder_id']]
        if decoder_filter and decoder_filter.casefold() not in (decoder['id']+' '+decoder['name']).casefold():
            continue
        matched.append(row)
        rate = row['logical_error_per_round']
        rate_text = f'{rate:.6g}' if rate is not None else '—'
        if row.get('logical_error_uncertainty') is not None:
            rate_text += f" ± {row['logical_error_uncertainty']:.3g}"
        speed = row['speed']
        timing = (f"{speed['relation']} {speed['value']:g}; {speed['metric']}" if speed else '—')
        source = sources[row['source_id']]
        lines.append(f"| {decoder['name']} | {row['evidence']}; {row['workload']} | {rate_text} | {timing} | [{row['source_id']}]({source['url']}) |")
    if not matched:
        raise ValueError('No literature decoder matches that filter')
    # Include configuration caveats next to the table rather than implying paired accuracy/speed.
    lines += ['', '### Benchmark qualifications', '']
    for row in matched:
        lines.append(f"- **{row['id']}** ({row['source_locator']}): {row['notes']}")
    lines += ['', '### Integration status', '']
    for decoder in decoders.values():
        if any(row['decoder_id'] == decoder['id'] for row in matched):
            lines.append(f"- **{decoder['name']}**: {decoder['integration_status']}")
    lines += ['', 'Literature observations remain separate from local simulation and IBM per-shot results. '
        'Do not divide our seven-round IBM failure probability by seven to create a literature-comparable rate.', '']
    return '\n'.join(lines)


def main():
    """Load frozen surface-code cases and save a reproducible comparison with input fingerprints."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='runs/surface-d7-baseline/sweep')
    p.add_argument('--out', default='runs/pymatching2-comparison')
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--latency-shots', type=int, default=1000)
    p.add_argument('--sweep-key', help='Read an appended sweep and append comparison to existing files')
    p.add_argument('--literature', action='store_true', help='Read the saved decoder literature database; execute no circuits')
    p.add_argument('--literature-decoder', help='Filter literature by decoder name or identifier')
    p.add_argument('--literature-json', action='store_true', help='Print literature JSON without embedded PDF bytes')
    p.add_argument('--sycamore-case', help='Benchmark one surface-code configuration in the local Sycamore ZIP')
    p.add_argument('--sycamore-shots', type=int, help='Size of a shared random subset, default all shots')
    p.add_argument('--sycamore-seed', type=int, default=42, help='Seed for selecting shared Sycamore shots')
    args = p.parse_args()
    if args.sycamore_case:
        path = Path(args.out)/'comparison.json'
        if not path.is_file():
            p.error('Sycamore mode requires the existing comparison JSON in --out')
        result = compare_sycamore(args.sycamore_case, args.sycamore_shots, args.sycamore_seed, args.repeats)
        document = json.loads(path.read_text(encoding='utf-8'))
        dataset = document['literature_database']['datasets']['sycamore-memory-6804040']
        key = f"{args.sycamore_case}-n{result['shots']}-seed{args.sycamore_seed}"
        dataset.setdefault('local_same_model_benchmarks', {})[key] = result
        write_json(path, document)
        if result['exact_contraction_skipped']:
            print('Exact tensor contraction:', result['exact_contraction_skipped'])
        return
    if args.literature or args.literature_decoder or args.literature_json:
        # Resolve the established result path from this script, independent of the working directory.
        database = json.loads((Path(__file__).resolve().parent/'runs/pymatching2-comparison/comparison.json').read_text(encoding='utf-8'))['literature_database']
        validate_literature(database)
        if args.literature_json:
            # Keep the human-facing export compact; original PDF bytes remain in the stored database.
            database['paper_archives'] = {key: {k: v for k, v in value.items() if k != 'data'}
                for key, value in database.get('paper_archives', {}).items()}
            # The CLI exports metadata. The original sample payload remains available in the saved JSON.
            subset = database.get('datasets', {}).get('gdm-qec-release', {}).get('d7_original_subset', {})
            subset.pop('gzip_tfrecord_base64', None)
            if args.literature_decoder:
                ids = {d['id'] for d in database['decoders'] if args.literature_decoder.casefold() in (d['id']+' '+d['name']).casefold()}
                if not ids:
                    p.error('No literature decoder matches that filter')
                database['decoders'] = [d for d in database['decoders'] if d['id'] in ids]
                database['benchmarks'] = [b for b in database['benchmarks'] if b['decoder_id'] in ids]
            print(json.dumps(database, indent=2, ensure_ascii=True))
        else:
            try:
                # Escape characters unavailable in older Windows consoles without losing JSON fidelity.
                text = literature_markdown(database, args.literature_decoder)
                print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            except ValueError as exc:
                p.error(str(exc))
        return
    if args.repeats < 3 or args.latency_shots < 1:
        p.error('Use repeats >= 3 and latency-shots >= 1')
    source, out = Path(args.input), Path(args.out)
    if args.sweep_key:
        # Use exactly the saved new-p shots, retaining earlier timing and accuracy data.
        from hardware_trial import decode_measurements
        saved = json.loads((source/'benchmark.json').read_text())['additional_sweeps'][args.sweep_key]
        path = out/'comparison.json'
        if not path.is_file() or not (out/'report.md').is_file():
            p.error('Appended comparisons require existing JSON and Markdown files')
        previous = json.loads(path.read_text())
        result = {'versions': versions(), 'python_executable': sys.executable,
                  'platform': platform.platform(), 'parameters': vars(args),
                  'source_parameters': saved['parameters'], 'cases': []}
        for record in saved['cases']:
            c = stim.Circuit(record['circuit'])
            m = decode_measurements(record['measurements'])
            case = compare_case(c, m, args.repeats, args.latency_shots)
            case.update(basis=record['basis'], rounds=record['rounds'], noise=record['noise'],
                        circuit_sha256=hashlib.sha256(str(c).encode()).hexdigest())
            result['cases'].append(case)
            print(record['id'], [(v['algorithm'], v['failures']) for v in case['variants']], flush=True)
        previous.setdefault('additional_comparisons', {})[args.sweep_key] = result
        write_json(path, previous)
        text = (out/'report.md').read_text()
        marker = f'\n<!-- appended:{args.sweep_key} -->\n'
        text = text.split(marker)[0]
        noise = saved['parameters']['noise']
        section = report_markdown(result).replace('noise parameter 0.005', f'noise parameter {noise}')
        (out/'report.md').write_text(text+marker+section, encoding='utf-8')
        return
    files = sorted(source.glob('surface-d7-r*-*.stim'))
    if not files:
        p.error('No saved distance-7 surface-code cases found')
    out.mkdir(parents=True, exist_ok=False)
    result = {'versions': versions(), 'python_executable': sys.executable,
              'platform': platform.platform(), 'parameters': vars(args), 'cases': []}
    for path in files:
        c = stim.Circuit.from_file(str(path))
        measurements_path = source/f'{path.stem}-measurements.npy'
        m = np.load(measurements_path, allow_pickle=False)
        case = compare_case(c, m, args.repeats, args.latency_shots)
        case.update(basis=path.stem.rsplit('-',1)[1], rounds=int(path.stem.split('-r')[1].split('-')[0]),
            circuit_path=str(path.resolve()), circuit_sha256=hashlib.sha256(str(c).encode()).hexdigest(),
            measurements_path=str(measurements_path.resolve()),
            measurements_sha256=hashlib.sha256(measurements_path.read_bytes()).hexdigest())
        result['cases'].append(case)
        write_json(out/'comparison.json', result)
        (out/'report.md').write_text(report_markdown(result), encoding='utf-8')
        print(path.stem, [(v['algorithm'],v['bit_packed'],v['failures'],round(v['batch_shots_per_second'])) for v in case['variants']],flush=True)


if __name__ == '__main__':
    main()
