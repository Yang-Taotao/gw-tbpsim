"""
Config file
===========

Configurations for:

1) JAX settings: jax_settings()
2) Static initial values:
    - PARAM_BASE: Initial param [m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination].
    - F_BASE: Frequency settings for signal and reference.
    - M_BASE: Component mass settings for mass entries, same for m1, m2.
"""
# Import
import os
import jax

# JAX settings
# =========================================================================== #


def jax_settings() -> None:
    """Initiate JAX settings"""
    # Setting - Enable float64 precision -> needed for waveform normalization
    jax.config.update("jax_enable_x64", True)
    # Setting - Manual memory allocation -> set to true if OOM occurs
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Setting - Set up presistent cache
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_compilation_cache_dir", ".jaxcache")
    # Print
    print("|== JAX config initialized ==|")


# =========================================================================== #

# Initial CONST config
# =========================================================================== #
# Freq - min, max, step size
F_BASE = 24.0, 512.0, 0.5
# Mass - min, max, step size
M_BASE = 1.0, 21.0, 100
# Param base - m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination
PARAM_BASE = [36.0, 29.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0]
# =========================================================================== #
