"""
Core methods.
"""
# Imports
import jax
import jax.numpy as jnp
import ripple
from ripple.waveforms import IMRPhenomXAS
# Import constants
from src.gw_tbpsim.constant import F_SIG, F_REF, F_PSD, F_DIFF

# Func


def inner_prod(vec_a: jnp.ndarray, vec_b: jnp.ndarray) -> float:
    """Noise weighted inner product between vectors a and b.

    Args:
        vec_a (jnp.ndarray): Vector a.
        vec_b (jnp.ndarray): Vector b.

    Returns:
        float: Noise weighted inner product.
    """
    # Get components
    numerator = jnp.abs(vec_a.conj() * vec_b)
    integrand = numerator / F_PSD
    # Return one side noise weighted inner products
    return 4 * F_DIFF * integrand.sum(axis=-1)


def waveform(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Waveform generation with ripple

    Args:
        theta (jnp.ndarray): GW param - mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple of GW strain hp, hc.
    """
    # Get hp, hc
    hp, hc = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(
        F_SIG, theta, F_REF)
    # Return
    return hp, hc


@jax.jit
def waveform_norm(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Waveform generation with ripple

    Args:
        theta (jnp.ndarray): GW param - mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Tuple of normalized GW strain hp, hc.
    """
    # Get waveform
    hp, hc = waveform(theta)
    # Normalized waveform - waveform/norm factor
    hp_norm = hp / jnp.sqrt(inner_prod(hp, hp))
    hc_norm = hc / jnp.sqrt(inner_prod(hc, hc))
    # Return
    return hp_norm, hc_norm
