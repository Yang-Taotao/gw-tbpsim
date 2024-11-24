"""
Main script file for testing purposes.
"""

# %%
# Imports
import os
import jax
import src.gw_tbpsim as gw
from src.gw_tbpsim.constant import THETA_ARRAY


# %%
# JAX settings
jax.config.update("jax_enable_x64", True)
# Setting - Manual memory allocation -> set to true if OOM occurs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# Setting - Set up presistent cache
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_dir", ".jaxcache")

# %%
# Test - det
det_hp = gw.log_sqrt_det_hp(THETA_ARRAY)
det_hc = gw.log_sqrt_det_hc(THETA_ARRAY)
print(det_hp, det_hc)
