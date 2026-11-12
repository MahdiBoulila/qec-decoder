"""Tests of PyMatching option equivalence and same-shot comparison accounting."""
import unittest
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
from compare_pymatching import compare_case, literature_markdown, validate_literature
from pipeline import benchmark, generated_memory


class DecoderOptionTests(unittest.TestCase):
    """Validate that representation and batch changes do not silently alter predictions."""
    def test_packed_and_single_shot_paths_match_and_correlated_cli_path(self):
        """Match packed, unpacked, batch, single-shot, and pipeline decoder paths on the same samples."""
        c = generated_memory(distance=3, rounds=3)
        m = c.compile_sampler(seed=7).sample(256)
        comparison = compare_case(c, m, repeats=3, latency_shots=16)
        self.assertEqual(len(comparison['variants']), 4)
        for name, decoder in [('ordinary', 'pymatching'), ('correlated', 'pymatching-correlated')]:
            expected = benchmark(c, m, decoder)['failures']
            for v in comparison['variants']:
                if v['algorithm'] == name:
                    self.assertEqual(v['failures'], expected)
        pair = comparison['paired_accuracy_comparison']
        self.assertEqual(pair['net_failures_avoided'],
            comparison['variants'][0]['failures']-comparison['variants'][2]['failures'])


class LiteratureDatabaseTests(unittest.TestCase):
    """Protect evidence provenance, missing metrics, the archived paper, and existing runs."""
    @classmethod
    def setUpClass(cls):
        """Load the actual persisted registry so schema checks exercise delivered data."""
        path = Path(__file__).resolve().parent/'runs/pymatching2-comparison/comparison.json'
        if not path.is_file():
            raise unittest.SkipTest('Requires the optional local decoder literature archive')
        cls.document = json.loads(path.read_text(encoding='utf-8'))
        cls.database = cls.document['literature_database']

    def test_registry_and_existing_results_integrity(self):
        """Verify citations and that adding literature did not alter any historical experiment."""
        validate_literature(self.database)
        experiments = {k: v for k, v in self.document.items() if k != 'literature_database'}
        digest = hashlib.sha256(json.dumps(experiments, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, self.database['existing_experiments_sha256'])

    def test_original_pdf_archive_is_lossless(self):
        """Decode the embedded original paper and verify format, length, and content hash."""
        archive = self.database['paper_archives']['alphaqubit-2024']
        pdf = gzip.decompress(base64.b64decode(archive['data'], validate=True))
        self.assertTrue(pdf.startswith(b'%PDF-'))
        self.assertIn(b'%%EOF', pdf[-2048:])
        self.assertEqual(len(pdf), archive['size_bytes'])
        self.assertEqual(hashlib.sha256(pdf).hexdigest(), archive['sha256'])

    def test_rendering_keeps_absent_rates_absent(self):
        """Check filtering and avoid inventing a zero logical-error rate for speed-only results."""
        text = literature_markdown(self.database, 'alphaqubit2-rt')
        self.assertIn('3.73e-06', text)
        self.assertIn('0.9652', text)
        self.assertNotIn('| Helios |', text)
        self.assertTrue(any(row['logical_error_per_round'] is None for row in self.database['benchmarks']))
        with self.assertRaisesRegex(ValueError, 'No literature decoder'):
            literature_markdown(self.database, 'nonexistent decoder')

    def test_invalid_reference_and_duplicate_are_rejected(self):
        """Reject ambiguous records and citation links that no longer resolve in the registry."""
        database = copy.deepcopy(self.database)
        database['benchmarks'][0]['source_id'] = 'unknown-source'
        with self.assertRaisesRegex(ValueError, 'Unknown benchmark citation'):
            validate_literature(database)
        database['benchmarks'][0]['source_id'] = self.database['benchmarks'][0]['source_id']
        database['benchmarks'].append(database['benchmarks'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_literature(database)


if __name__ == '__main__':
    unittest.main()
