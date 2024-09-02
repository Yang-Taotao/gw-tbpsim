'''
Main script file for testing purposes.
'''
# Imports
import src.gw_tbpsim
from src.gw_tbpsim.constant import THETA_ARRAY
# Import JAX settings
import src.gw_tbpsim.config
# Set JAX settings
src.gw_tbpsim.config.jax_settings()

# Test - wf gen - 1d
hp_test, hc_test = src.gw_tbpsim.waveform(THETA_ARRAY)
print(f"hp_test.shape: {hp_test.shape}")
print(f"hc_test.shape: {hc_test.shape}")
# Test - wf normalization - 1d
hp_norm, hc_norm = src.gw_tbpsim.waveform_norm(THETA_ARRAY)
print(f"hp_norm.shape: {hp_norm.shape}")
print(f"hc_norm.shape: {hc_norm.shape}")
