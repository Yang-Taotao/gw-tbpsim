"""
Config file
===========

Configurations for:

1) JAX settings: jax_settings()
2) Static initial values:
    - PARAM_BASE: Initial mock GW150914 parameters [m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination].
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
    # Float64 support option - set to false
    jax.config.update("jax_enable_x64", False)
    # Mem allocation - manual allocation set to false
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Use persistent cache
    jax.config.update("jax_compilation_cache_dir", "./.jaxcache")
    print('|== JAX settings applied ==|')


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
