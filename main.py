"""
Main script file for testing purposes.
"""

# Imports
import os
import jax
from src import gw_tbpsim as gw
from src.gw_tbpsim.config import THETA_ARRAY, MC_ARRAY, ETA_ARRAY, MC_MOCK, ETA_MOCK

# JAX settings
jax.config.update("jax_enable_x64", True)
# Setting - Manual memory allocation -> set to true if OOM occurs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# Setting - Set up presistent cache
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_dir", ".jaxcache")

# Mock compilation - (10, 10, 8) batch
THETA_ARRAY_MOCK = gw.theta_gen(THETA_ARRAY, MC_MOCK, ETA_MOCK)
log_den_hp_mock, log_den_hc_mock = gw.mock_compile(THETA_ARRAY_MOCK)
# Batched run
log_den_hp, log_den_hc = gw.log_density_batch(THETA_ARRAY, MC_ARRAY, ETA_ARRAY)

# Plot gen
gw.log_fim_contour(MC_ARRAY, ETA_ARRAY, log_den_hp, "hp")
gw.log_fim_contour(MC_ARRAY, ETA_ARRAY, log_den_hc, "hc")
