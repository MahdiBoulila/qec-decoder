"""Verify shot-count bounds and in-place submission using memory-only mocks.

No test contacts IBM or creates result files. The real submission remains an
explicit CLI operation after these checks pass.
"""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np
import stim
from qiskit import QuantumCircuit
import hardware_trial


class TrialSubmissionTests(unittest.TestCase):
    """Protect requested shot counts, duplicate prevention, and stored-data integrity."""

    def _run_submission(self, shots, phase='submit', duplicate=False):
        """Exercise the real CLI branches with network and filesystem effects mocked."""
        source = stim.Circuit('R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]')
        compiled = b'fixed-compiled-circuit'
        options = {'twirling': {'enable_gates': False, 'enable_measure': False},
                   'execution': {'init_qubits': True}, 'max_execution_time': 60}
        manifest = {'shots': shots if phase == 'submit' else 100,
            'verified_plan': 'open', 'instance': 'test-instance', 'backend': 'test-backend',
            'isa_depth': 1, 'isa_operations': {}, 'sampler_options': options,
            'isa_sha256': hashlib.sha256(compiled).hexdigest(),
            'circuit_sha256': hashlib.sha256(str(source).encode()).hexdigest()}
        if phase == 'submit-extra':
            manifest['job_id'] = 'original-100-shot-job'
        if duplicate:
            manifest['additional_experiments'] = {'shots1000': {'job_id': 'prior-job'}}
        svc = Mock()
        svc.backend.return_value.status.return_value.operational = True
        svc.instances.return_value = [{'crn': 'test-instance', 'plan': 'open'}]
        sampler = Mock()
        sampler.run.return_value.job_id.return_value = 'test-job'
        saved = []
        # Deep-copy writes so later dictionary changes cannot change earlier snapshots.
        def capture(path, value):
            """Record intended writes in memory for assertions, without creating files."""
            saved.append(copy.deepcopy(value))
        with patch('sys.argv', ['hardware_trial.py', phase, '--out', 'existing-run',
                                '--shots', str(shots)]), \
             patch.object(Path, 'read_text', return_value=json.dumps(manifest)), \
             patch.object(Path, 'read_bytes', return_value=compiled), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'open', mock_open()), \
             patch.object(Path, 'write_text'), \
             patch.object(hardware_trial.qpy, 'load', return_value=[QuantumCircuit(1, 1)]), \
             patch.object(hardware_trial, 'load', return_value=source), \
             patch.object(hardware_trial, 'write_json', side_effect=capture), \
             patch.object(hardware_trial, 'service', return_value=svc), \
             patch.object(hardware_trial, 'SamplerV2', return_value=sampler) as factory, \
             patch('builtins.print'):
            if duplicate:
                with self.assertRaises(ValueError):
                    hardware_trial.main()
                sampler.run.assert_not_called()
                return
            hardware_trial.main()
        self.assertEqual(sampler.run.call_args.kwargs, {'shots': shots})
        self.assertEqual(factory.call_args.kwargs['options'], options)
        if phase == 'submit-extra':
            self.assertEqual(saved[-1]['shots'], 100)  # Original experiment survives.
            record = saved[-1]['additional_experiments']['shots1000']
            self.assertEqual(record['shots'], 1000)
            self.assertEqual(record['job_id'], 'test-job')
        else:
            self.assertEqual(saved[-1]['shots'], shots)

    def test_exact_shot_counts_for_original_and_appended_jobs(self):
        """The original 100-shot and appended 1000-shot requests reach the API unchanged."""
        self._run_submission(100)
        self._run_submission(1000, phase='submit-extra')

    def test_duplicate_append_does_not_submit(self):
        """Reusing an existing added-experiment key must never create another job."""
        self._run_submission(1000, phase='submit-extra', duplicate=True)

    def test_lossless_measurement_payload(self):
        """Compressed in-JSON shots round-trip exactly and reject corrupted checksums."""
        m = np.random.default_rng(42).integers(0, 2, (1000, 385)).astype(bool)
        payload = hardware_trial.encode_measurements(m)
        np.testing.assert_array_equal(hardware_trial.decode_measurements(payload), m)
        payload['sha256'] = 'invalid'
        with self.assertRaises(ValueError):
            hardware_trial.decode_measurements(payload)

    def test_calibration_is_not_reported_as_observed_pauli_faults(self):
        """Provider coherence and assignment estimates must never become invented X/Y/Z truth."""
        props=Mock()
        props.qubits=[[],[]]
        props.gates=[]
        props.t1.return_value=0.0001
        props.t2.return_value=0.00012
        props.readout_error.return_value=0.02
        props.last_update_date='test-calibration'
        job=Mock()
        job.properties.return_value=props
        result=hardware_trial.hardware_calibration(job)
        self.assertAlmostEqual(result['median_t1_us'],100)
        self.assertAlmostEqual(result['median_t2_us'],120)
        self.assertEqual(result['pauli_fault_rates'],{'X':None,'Y':None,'Z':None})

    def test_shot_bound_rejected_before_connection(self):
        """Requests above the explicit experiment bound fail before account access."""
        with patch('sys.argv', ['hardware_trial.py', 'submit-extra', '--shots', '1001']), \
             patch.object(hardware_trial, 'service') as connect, \
             patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                hardware_trial.main()
        connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
