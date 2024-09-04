"""
Main script file for testing purposes.
"""

# Imports
import src.gw_tbpsim as gw
from src.gw_tbpsim.constant import THETA_ARRAY, F_SIG

# Test - wf gen - 1d
waveform_hp = gw.waveform_hp(THETA_ARRAY, F_SIG)
waveform_hc = gw.waveform_hc(THETA_ARRAY, F_SIG)
grad_hp = gw.grad_hp(THETA_ARRAY)
grad_hc = gw.grad_hc(THETA_ARRAY)
