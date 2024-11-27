"""
Configs.
========

Primary collection of configs.
    - PARAM_ARRAY: Initial mock GW150914 param 
        - [m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]
    - F_SIG: Frequency signal.
    - F_REF: Frequency reference.
    - F_DIFF: Frequency range. 
    - F_SAMP: Number of frequency samples.
    - F_DURA: Duration of every sample.
    - F_PSD: Power Spectrum Density array.
    - MC_ARRAY: Chirp mass array.
    - ETA_ARRAY: Symmetric mass ratio array.
    - MCS_MOCK: Chirp mass array for mock compilation.
    - ETAS_MOCK: Symmetric mass ratio array for mock compilation.
"""

# Imports
import jax
import jax.numpy as jnp
import bilby

# Initial config
# =========================================================================== #
# Freq - min, max, step size
F_BASE = 24.0, 512.0, 0.5
# Mass - Chirp mass - min, max, step size
MC_BASE = 1.0, 21.0, 100
# Mass - Symmetric mass ratio - min, max, step size
ETA_BASE = 0.05, 0.25, 100
# Mass - Chirp mass - min, max, step size
MC_MOCK_BASE = 1.0, 21.0, 10
# Mass - Symmetric mass ratio - min, max, step size
ETA_MOCK_BASE = 0.05, 0.25, 10
# Param base - m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination
PARAM_BASE = [36.0, 29.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0]
# Theta base - mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
THETA_BASE = [28.0956, 0.2471, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0]
# =========================================================================== #
# PSD


def freq_psd(f_min: float, f_max: float, f_del: float) -> jax.Array:
    """Build Power Spectral Density array

    Args:
        f_min (float): Frequency minimum.
        f_max (float): Frequency maximum.
        f_del (float): Frequency step size.

    Returns:
        jax.Array: PSD.
    """
    # Get detector
    detector = bilby.gw.detector.get_empty_interferometer("H1")
    # Set sampling frequency
    detector.sampling_frequency = (f_max - f_min) / f_del
    # Set detector duration
    detector.duration = 1 / f_del
    # Return PSD
    return detector.power_spectral_density_array[1:]


# Freq related constants
F_SIG = jnp.arange(*F_BASE)
F_REF = F_BASE[0]
F_DIFF = F_BASE[1] - F_BASE[0]
F_SAMP = (F_BASE[1] - F_BASE[0]) / F_BASE[2]
F_DURA = 1 / F_BASE[2]
F_PSD = freq_psd(*F_BASE)

# Parameter arrays
MC_ARRAY = jnp.linspace(*MC_BASE, dtype=jnp.float64)
ETA_ARRAY = jnp.linspace(*ETA_BASE, dtype=jnp.float64)
PARAM_ARRAY = jnp.array(PARAM_BASE, dtype=jnp.float64)
THETA_ARRAY = jnp.array(THETA_BASE, dtype=jnp.float64)
MC_MOCK = jnp.linspace(*MC_MOCK_BASE, dtype=jnp.float64)
ETA_MOCK = jnp.linspace(*ETA_MOCK_BASE, dtype=jnp.float64)
