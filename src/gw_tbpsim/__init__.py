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
    # Get integrand - (vec_a.conj() * vec_b).real / F_PSD
    integrand = (vec_a.conj() * vec_b).real / F_PSD
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
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.real


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
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.imag


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
    _, wf = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.real


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
    _, wf = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.imag


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
    # Func return - complex128 dtype necessary
    return jnp.complex128(grad_hp_real + grad_hp_imag * 1j)


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
    # Func return - complex128 dtype necessary
    return jnp.complex128(grad_hc_real + grad_hc_imag * 1j)


# WIP
# =========================================================================== #
# FIM - Main ==> log.sqrt.det.FIM


def log_sqrt_det_hp(theta: jnp.ndarray):
    """
    Return the log based square root of the determinant of
    Fisher matrix projected onto the mc, eta space
    for hp waveform results
    """
    # Calculation
    # try:
    data_fim = projected_fim_hp(theta)
    # except AssertionError:
    #    data_fim = jnp.nan
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(data_fim)))


def log_sqrt_det_hc(theta: jnp.ndarray):
    """
    Return the log based square root of the determinant of
    Fisher matrix projected onto the mc, eta space
    for hc waveform results
    """
    # Calculation
    # try:
    data_fim = projected_fim_hc(theta)
    # except AssertionError:
    #    data_fim = jnp.nan
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(data_fim)))


# FIM projection sub func


def fim_phic(full_fim: jnp.ndarray, nd_val: int):
    """
    Calculate the conditioned matrix projected onto coalecense phase
    """
    # Equation 16 from Dent & Veitch
    fim_result = jnp.array(
        [
            full_fim[i, j] - full_fim[i, -1] * full_fim[-1, j] / full_fim[-1, -1]
            for i in range(nd_val - 1)
            for j in range(nd_val - 1)
        ]
    ).reshape([nd_val - 1, nd_val - 1])
    # Func return
    return fim_result


def fim_tc(gamma: jnp.ndarray, nd_val: int):
    """
    Project the conditional matrix back onto coalecense time
    """
    # Equation 18 Dent & Veitch
    fim_result = jnp.array(
        [
            gamma[i, j] - gamma[i, -1] * gamma[-1, j] / gamma[-1, -1]
            for i in range(nd_val - 2)
            for j in range(nd_val - 2)
        ]
    ).reshape([nd_val - 2, nd_val - 2])
    # Func return
    return fim_result


# FIM packers - checked


def fim_base(grads: jnp.ndarray):
    """
    Basic FIM entry packer
    """
    # Get parameter shape as nd_val
    nd_val = grads.shape[-1]

    # FIM entry calculator
    def fim_entry(i: int, j: int) -> float:
        """Calculate FIM entry with one side noise weighted inner product"""
        return inner_prod(grads[:, i], grads[:, j])

    # Get FIM index arrays
    idx_i, idx_j = jnp.triu_indices(nd_val)
    # Calculate FIM entries of upper half
    entries = jax.vmap(fim_entry)(idx_i, idx_j)
    # Construct temporary FIM
    fim_temp = jnp.zeros((nd_val, nd_val))
    # Populate upper trig with entries
    fim_temp = fim_temp.at[idx_i, idx_j].set(entries)
    # Populate lower trig with entries flipped
    fim_result = fim_temp + jnp.triu(fim_temp, k=1).T
    # Func return
    return fim_result


def fim_hp(thetas: jnp.ndarray):
    """
    Returns the fisher information matrix
    at a general value of mc, eta, tc, phic
    for hp waveform

    Args:
        thetas (array): [Mc, eta, t_c, phi_c]. Shape 1x4
    """
    # Generate the waveform derivatives
    grads = grad_hp(thetas)
    # Get FIM result
    fim_result = fim_base(grads)
    # Func return
    return fim_result


def fim_hc(thetas: jnp.ndarray):
    """
    Returns the fisher information matrix
    at a general value of mc, eta, tc, phic
    for hc waveform

    Args:
        thetas (array): [Mc, eta, t_c, phi_c]. Shape 1x4
    """
    # Generate the waveform derivatives
    grads = grad_hc(thetas)
    # Get FIM result
    fim_result = fim_base(grads)
    # Func return
    return fim_result
