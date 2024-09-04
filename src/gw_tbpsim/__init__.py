"""
Core methods.
"""

# Imports
# =========================================================================== #
import os
import jax
import jax.numpy as jnp
from ripple.waveforms import IMRPhenomXAS

# Import constants
from src.gw_tbpsim.constant import F_SIG, F_REF, F_PSD, F_DIFF

# JAX settings
jax.config.update("jax_enable_x64", True)
# Setting - Manual memory allocation -> set to true if OOM occurs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# Setting - Set up presistent cache
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_dir", ".jaxcache")

# Core Func
# =========================================================================== #


def inner_prod(vec_a: jax.Array, vec_b: jax.Array) -> jax.Array:
    """
    Noise weighted inner product between some vectors a and b.

    Args:
        vec_a (jax.Array): Vector a.
        vec_b (jax.Array): Vector b.

    Returns:
        jax.Array: One side noise weighted inner product.
    """
    # Get integrand - jnp.abs(vec_a.conj() * vec_b) / F_PSD
    integrand = jnp.abs(vec_a.conj() * vec_b) / F_PSD
    # Return one side noise weighted inner products
    return 4 * F_DIFF * integrand.sum(axis=-1)


# Waveform Gen
# =========================================================================== #


def waveform(theta: jax.Array, sig: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Normalized hp, hc. GW waveform generation with ripple

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        sig (jax.Array): Signal frequencies array

    Returns:
        tuple[jax.Array, jax.Array]: Tuple of normalized GW strain hp, hc
    """
    # Get ripple waveform
    hp, hc = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(sig, theta, F_REF)
    # Normalize ripple waveform - waveform / norm factor
    hp_norm = hp / jnp.sqrt(inner_prod(hp, hp))
    hc_norm = hc / jnp.sqrt(inner_prod(hc, hc))
    # Return
    return hp_norm, hc_norm


def waveform_hp(theta: jax.Array, sig: jax.Array) -> jax.Array:
    """
    Normalized hp. GW waveform generation with ripple

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        sig (jax.Array): Signal frequencies array

    Returns:
        jax.Array: Normalized GW strain hp
    """
    # Set to correct ripple theta dtype
    theta = jnp.float32(theta)
    # Get plus polarized GW strain
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(sig, theta, F_REF)
    # Return normalized waveform - wf / normaliztion factor
    return wf / jnp.sqrt(inner_prod(wf, wf))


def waveform_hc(theta: jax.Array, sig: jax.Array) -> jax.Array:
    """
    Normalized hc. GW waveform generation with ripple

    Args:
        theta (jnp.ndarray): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        sig (jnp.ndarray): Signal frequencies array

    Returns:
        hc_norm (jnp.ndarray): Normalized GW strain hc
    """
    # Set to correct ripple theta dtype
    theta = jnp.float32(theta)
    # Get plus polarized GW strain
    _, wf = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(sig, theta, F_REF)
    # Return normalized waveform - wf / normaliztion factor
    return wf / jnp.sqrt(inner_prod(wf, wf))


# Grad calc
# =========================================================================== #


def grad_hp(theta: jax.Array) -> jax.Array:
    """
    Gradients of hp against GW waveform.
    Mapped to signal frequencies for plus polarizations

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination

    Returns:
        jax.Array: Mapped gradients
            d(hp) / d(thetas)
    """
    # Set to correct gradient theta dtype -> d(hp)/d(thetas)
    thetas = jnp.complex128(theta)
    # Return gradients mapped to signal frequencies
    return jax.vmap(
        jax.grad(waveform_hp, holomorphic=True),
        in_axes=(None, 0),
    )(thetas, F_SIG)


def grad_hc(theta: jax.Array) -> jax.Array:
    """
    Gradients of hc against GW waveform.
    Mapped to signal frequencies for cross polarizations

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination

    Returns:
        jax.Array: Mapped gradients
            d(hc) / d(thetas)
    """
    # Set to correct gradient theta dtype -> d(hc)/d(thetas)
    thetas = jnp.complex128(theta)
    # Return gradients mapped to signal frequencies
    return jax.vmap(
        jax.grad(waveform_hc, holomorphic=True),
        in_axes=(None, 0),
    )(thetas, F_SIG)
