"""
Constant
========

Primary collection of constant values, static and calculated ones.
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
"""

# Imports
import jax
import jax.numpy as jnp
import bilby
from .config import F_BASE, MC_BASE, ETA_BASE, PARAM_BASE, THETA_BASE

# Func


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


# Calcuated CONST

F_SIG = jnp.arange(*F_BASE)
F_REF = F_BASE[0]
F_DIFF = F_BASE[1] - F_BASE[0]
F_SAMP = (F_BASE[1] - F_BASE[0]) / F_BASE[2]
F_DURA = 1 / F_BASE[2]
F_PSD = freq_psd(*F_BASE)
MC_ARRAY = jnp.linspace(*MC_BASE)
ETA_ARRAY = jnp.linspace(*ETA_BASE)
PARAM_ARRAY = jnp.array(PARAM_BASE)
THETA_ARRAY = jnp.array(THETA_BASE)
