"""
Main script file for testing purposes.
"""

# Imports
import os
import jax
import src.gw_tbpsim as gw
from src.gw_tbpsim.constant import THETA_ARRAY, F_SIG

# JAX settings
jax.config.update("jax_enable_x64", True)
# Setting - Manual memory allocation -> set to true if OOM occurs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# Setting - Set up presistent cache
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_dir", ".jaxcache")

# Test - wf gen - 1d
waveform_hp = gw.waveform_hp(THETA_ARRAY, F_SIG)
waveform_hc = gw.waveform_hc(THETA_ARRAY, F_SIG)
grad_hp = gw.grad_hp(THETA_ARRAY)
grad_hc = gw.grad_hc(THETA_ARRAY)
