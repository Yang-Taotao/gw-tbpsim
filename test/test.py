"""
Main test file.
"""

# Imports
import unittest
import jax.numpy as jnp
import src.gw_tbpsim as gw
from src.gw_tbpsim.constant import THETA_ARRAY, F_SIG


# Tester
class TestGW(unittest.TestCase):
    """
    Test GW waveform generation and gradients calculation
    """

    def test_waveform_hp(self):
        """
        Check hp waveform shape
        """
        wf = gw.waveform_hp(THETA_ARRAY, F_SIG)
        self.assertEqual(wf.shape, F_SIG.shape)

    def test_waveform_hc(self):
        """
        Check hc waveform shape
        """
        wf = gw.waveform_hc(THETA_ARRAY, F_SIG)
        self.assertEqual(wf.shape, F_SIG.shape)

    def test_waveform_hp_valid(self):
        """
        Check if hp waveform contains NaN
        """
        wf = gw.waveform_hp(THETA_ARRAY, F_SIG)
        self.assertFalse(jnp.isnan(wf).any())

    def test_waveform_hc_valid(self):
        """
        Check if hc waveform contains NaN
        """
        wf = gw.waveform_hp(THETA_ARRAY, F_SIG)
        self.assertFalse(jnp.isnan(wf).any())

    def test_grad_hp(self):
        """
        Check hp gradients shape
        """
        grads = gw.grad_hp(THETA_ARRAY)
        self.assertEqual(grads.shape, (F_SIG.shape[0], THETA_ARRAY.shape[0]))

    def test_grad_hc(self):
        """
        Check hc gradients shape
        """
        grads = gw.grad_hc(THETA_ARRAY)
        self.assertEqual(grads.shape, (F_SIG.shape[0], THETA_ARRAY.shape[0]))

    def test_grad_hp_valid(self):
        """
        Check if hp gradients contain NaN
        """
        grads = gw.grad_hp(THETA_ARRAY)
        self.assertFalse(jnp.isnan(grads).any())

    def test_grad_hc_valid(self):
        """
        Check if hc gradients contain NaN
        """
        grads = gw.grad_hc(THETA_ARRAY)
        self.assertFalse(jnp.isnan(grads).any())


if __name__ == "__main__":
    unittest.main()
