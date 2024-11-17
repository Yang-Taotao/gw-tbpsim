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


# Waveform Func
# =========================================================================== #


def hp_real(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hp waveform, real part.

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        f_sig (jax.Array): Signal frequencies array

    Returns:
        jax.Array: The real part of normalized hp waveform
    """
    # Get hp waveform with Ripple
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(jnp.array([f_sig]), theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.real[0]


def hp_imag(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hp waveform, imaginary part.

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        f_sig (jax.Array): Signal frequencies array

    Returns:
        jax.Array: The imaginary part of normalized hp waveform
    """
    # Get hp waveform with Ripple
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(jnp.array([f_sig]), theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.imag[0]


def hc_real(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hc waveform, real part.

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        f_sig (jax.Array): Signal frequencies array

    Returns:
        jax.Array: The real part of normalized hc waveform
    """
    # Get hc waveform with Ripple
    _, wf = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(jnp.array([f_sig]), theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.real[0]


def hc_imag(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hc waveform, imaginary part.

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination
        f_sig (jax.Array): Signal frequencies array

    Returns:
        jax.Array: The imaginary part of normalized hc waveform
    """
    # Get hc waveform with Ripple
    _, wf = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(jnp.array([f_sig]), theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.imag[0]


# Gradiant Func
# =========================================================================== #


def grad_hp(theta: jax.Array) -> jax.Array:
    """
    Gradients of normalized hp against hp parameters.

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination

    Returns:
        jax.Array: Mapped gradients
            d(hp) / d(thetas)
    """
    # Map gradiant result
    grad_hp_real = jax.vmap(jax.grad(hp_real), in_axes=(None, 0))(theta, F_SIG)
    grad_hp_imag = jax.vmap(jax.grad(hp_imag), in_axes=(None, 0))(theta, F_SIG)
    # Func return
    return grad_hp_real + grad_hp_imag * 1j


def grad_hc(theta: jax.Array) -> jax.Array:
    """
    Gradients of normalized hp against hp parameters.

    Args:
        theta (jax.Array): GW param
            mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination

    Returns:
        jax.Array: Mapped gradients
            d(hp) / d(thetas)
    """

    # Map gradiant result
    grad_hc_real = jax.vmap(jax.grad(hc_real), in_axes=(None, 0))(theta, F_SIG)
    grad_hc_imag = jax.vmap(jax.grad(hc_imag), in_axes=(None, 0))(theta, F_SIG)
    # Func return
    return grad_hc_real + grad_hc_imag * 1j
