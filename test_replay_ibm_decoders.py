"""Tests for the bounded exact decoder; these tests never contact IBM."""
import itertools
import unittest
import numpy as np
import stim
from replay_ibm_decoders import ExactDemDecoder


class ExactDecoderTests(unittest.TestCase):
    """Validate exact DEM inference against independent brute-force probability enumeration."""
    def test_sum_over_logical_classes_matches_brute_force(self):
        """Confirm class probability sums, rather than a largest-single-fault shortcut, determine the result."""
        model = stim.DetectorErrorModel('error(0.2) D0 L0\nerror(0.15) D0\nerror(0.15) D0')
        decoder = ExactDemDecoder(model)
        joint = np.zeros((2,2))
        for faults in itertools.product([0,1],repeat=3):
            probability = 1
            for occurred,p in zip(faults,[0.2,0.15,0.15]):
                probability *= p if occurred else 1-p
            joint[faults[0],sum(faults)%2] += probability
        np.testing.assert_allclose(decoder.joint,joint)
        # Most-probable logical class differs from the largest single fault prior.
        self.assertEqual(decoder.decode_batch(np.array([[1]],dtype=bool))[0,0],0)

    def test_hyperedge_separator_preserves_single_fault(self):
        """Ensure components of one separated DEM fault remain a correlated single event."""
        model=stim.DetectorErrorModel('error(0.1) D0 L0 ^ D1')
        decoder=ExactDemDecoder(model)
        self.assertAlmostEqual(decoder.joint[1,3],0.1)
        self.assertEqual(decoder.decode_batch(np.array([[1,1]],dtype=bool))[0,0],1)

    def test_size_guard(self):
        """Reject an exponentially large lookup before allocating the state distribution."""
        with self.assertRaises(ValueError):
            ExactDemDecoder(stim.DetectorErrorModel('detector D20'))


if __name__ == '__main__':
    unittest.main()
