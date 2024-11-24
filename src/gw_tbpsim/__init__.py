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


def inner_prod(array_a: jax.Array, array_b: jax.Array) -> jax.Array:
    """
    Noise weighted inner product between array array_a and array_b.

    Args:
        array_a (jax.Array): Array a.
        array_b (jax.Array): Array b.

    Returns:
        jax.Array: One side noise weighted inner product.
    """
    # Get integrand
    integrand = (array_a.conj() * array_b).real / F_PSD
    # Return one side noise weighted inner products
    return 4 * F_DIFF * integrand.sum(axis=-1)


# Waveform Func
# =========================================================================== #


def hp_real(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hp waveform, hp.real.

    Args:
        theta (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
        f_sig (jax.Array): Signal frequencies array.

    Returns:
        jax.Array: The real part of normalized hp waveform.
    """
    # Get hp waveform with Ripple
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.real


def hp_imag(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hp waveform, hp.imag.

    Args:
        theta (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
        f_sig (jax.Array): Signal frequencies array.

    Returns:
        jax.Array: The imaginary part of normalized hp waveform.
    """
    # Get hp waveform with Ripple
    wf, _ = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.imag


def hc_real(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hc waveform, hc.real.

    Args:
        theta (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
        f_sig (jax.Array): Signal frequencies array.

    Returns:
        jax.Array: The real part of normalized hc waveform.
    """
    # Get hc waveform with Ripple
    _, wf = IMRPhenomXAS.gen_IMRPhenomXAS_hphc(f_sig, theta, F_REF)
    # Calculate normalized waveform
    wf_norm = wf / jnp.sqrt(inner_prod(wf, wf))
    # Func return
    return wf_norm.real


def hc_imag(theta: jax.Array, f_sig: jax.Array) -> jax.Array:
    """
    Normalized hc waveform, hc.imag.

    Args:
        theta (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
        f_sig (jax.Array): Signal frequencies array.

    Returns:
        jax.Array: The imaginary part of normalized hc waveform.
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
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jax.Array: Mapped gradients
            evaluated as d(hp) / d(thetas).
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
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jax.Array: Mapped gradients
            evaluated as d(hp) / d(thetas).
    """

    # Map gradiant result
    grad_hc_real = jax.vmap(jax.grad(hc_real), in_axes=(None, 0))(theta, F_SIG)
    grad_hc_imag = jax.vmap(jax.grad(hc_imag), in_axes=(None, 0))(theta, F_SIG)
    # Func return - complex128 dtype necessary
    return jnp.complex128(grad_hc_real + grad_hc_imag * 1j)


# WIP
# =========================================================================== #
# FIM - Main ==> log.sqrt.det.FIM => density statictics


def log_sqrt_det_hp(theta: jnp.Array) -> jnp.Array:
    """
    Calculate intermediate statistics for density building
    with log.sqrt.det.Metric on hp waveform based results.

    Args:
        theta (jnp.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jnp.Array: intermediate statistics
            as log.sqrt.det.Metric.
    """
    # Calculation
    metric = projected_fim_hp(theta)
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(metric)))


def log_sqrt_det_hc(theta: jnp.Array) -> jnp.Array:
    """
    Calculate intermediate statistics for density building
    with log.sqrt.det.Metric on hc waveform based results.

    Args:
        theta (jnp.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jnp.Array: intermediate statistics
            as log.sqrt.det.Metric.
    """
    # Calculation
    metric = projected_fim_hc(theta)
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(metric)))


# %%
# FIM - Projected and simple FIM


def projected_fim_hp(thetas: jnp.Array) -> jnp.Array:
    """
    Projected Fisher Information Matrix function call for hp waveforms.

    Args:
        thetas (jnp.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jnp.Array: Projected metric on mc, eta space for hp waveforms.
    """
    # Get full FIM and dimensions
    full_fim = fim_hp(thetas)
    # Calculate the conditioned matrix for phase
    gamma = fim_phic(full_fim)
    # Calculate the conditioned matrix for time
    metric = fim_tc(gamma)
    # Func return
    return metric


def projected_fim_hc(thetas: jnp.Array) -> jnp.Array:
    """
    Projected Fisher Information Matrix function call for hc waveforms.

    Args:
        thetas (jnp.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jnp.Array: Projected metric on mc, eta space for hc waveforms.
    """
    # Get full FIM and dimensions
    full_fim = fim_hc(thetas)
    # Calculate the conditioned matrix for phase
    gamma = fim_phic(full_fim)
    # Calculate the conditioned matrix for time
    metric = fim_tc(gamma)
    # Func return
    return metric


# FIM - projection sub func


def fim_phic(fim: jnp.ndarray):
    """
    Project Fisher Information Matrix onto phic with Eq. 16 of 1311.7174.

    Args:
        fim (jnp.ndarray): Fisher Information Matrix.

    Returns:
        jnp.Array: projected conditional matrix gamma.
    """
    # Local repo
    nd_val = fim.shape[-1]
    idx_i = jnp.arange(nd_val - 1)
    idx_j = jnp.arange(nd_val - 1)
    last_entry = fim[-1, -1]

    # Eq. 16 - Dent & Veitch
    def dv_16(i: int, j: int) -> float:
        """
        Conditional matrix entry calculation for Eq. 16 of 1311.7174.

        Args:
            i (int): index i.
            j (int): index j.

        Returns:
            float: projected conditional matrix entry of gamma.
        """
        # Calcualte offset entry
        offset = fim[i, -1] * fim[-1, j] / last_entry
        # Get entry result with conditional offset - prevent div by 0
        entry = jnp.where(last_entry != 0, fim[i, j] - offset, fim[i, j])
        # Func return
        return entry

    # Build gamma result
    gamma = jax.vmap(jax.vmap(dv_16, in_axes=(None, 0)), in_axes=(0, None))(
        idx_i, idx_j
    )
    # Func return
    return gamma


def fim_tc(gamma: jnp.Array) -> jnp.Array:
    """
    Project conditional matrix back onto tc with Eq. 18 of 1311.7174.

    Args:
        gamma (jnp.ndarray): conditional matrix gamma.

    Returns:
        jnp.Array: projected metric.
    """
    # Local repo
    nd_val = gamma.shape[-1]
    idx_i = jnp.arange(nd_val - 1)
    idx_j = jnp.arange(nd_val - 1)
    last_entry = gamma[-1, -1]

    # Eq. 18 - Dent & Veitch
    def dv_18(i: int, j: int) -> float:
        """
        Metric entry calculation for Eq. 18 of 1311.7174.

        Args:
            i (int): index i.
            j (int): index j.

        Returns:
            float: projected metric entry.
        """
        # Calculate offset entry
        offset = gamma[i, -1] * gamma[-1, j] / gamma[-1, -1]
        # Get entry result with conditional offset - prevent div by 0
        entry = jnp.where(last_entry != 0, gamma[i, j] - offset, gamma[i, j])
        # Func return
        return entry

    # Build metric result
    metric = jax.vmap(jax.vmap(dv_18, in_axes=(None, 0)), in_axes=(0, None))(
        idx_i, idx_j
    )
    # Func return
    return metric


# FIM builder


def fim_core(grads: jnp.Array) -> jnp.Array:
    """
    Fisher Information Matrix builder.

    Args:
        grads (jnp.Array): GW waveform gradients
            with shape (F_SIG.shape[0], THETA_ARRAY.shape[-1]).

    Returns:
        jnp.Array: Fisher Information Matrix.
    """
    # Get parameter shape as nd_val
    nd_val = grads.shape[-1]
    # Get FIM index arrays
    idx_i, idx_j = jnp.triu_indices(nd_val)

    # FIM entry calculator
    def fim_entry(i: int, j: int) -> float:
        """
        Calculate FIM entry with one side noise weighted inner product.

        Args:
            i (int): index i.
            j (int): index j.

        Returns:
            float: FIM entry at (i, j) index.
        """
        # Return
        return inner_prod(grads[:, i], grads[:, j])

    # Calculate FIM entries of upper half
    entries = jax.vmap(fim_entry)(idx_i, idx_j)
    # Construct temporary FIM
    fim_temp = jnp.zeros((nd_val, nd_val))
    # Populate upper trig with entries
    fim_temp = fim_temp.at[idx_i, idx_j].set(entries)
    # Populate lower trig with entries flipped
    fim = fim_temp + jnp.triu(fim_temp, k=1).T
    # Func return
    return fim


def fim_hp(thetas: jnp.Array) -> jnp.Array:
    """
    Build Fisher Information Matrix for hp waveform.

    Args:
        thetas (jnp.Array): GW param
            [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination]

    Returns:
        jnp.Array: Fisher Information Matrix of hp waveform
    """
    # Generate the waveform derivatives
    grads = grad_hp(thetas)
    # Return FIM result
    return fim_core(grads)


def fim_hc(thetas: jnp.Array) -> jnp.Array:
    """
    Build Fisher Information Matrix for hc waveform.

    Args:
        thetas (jnp.Array): GW param
            [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination]

    Returns:
        jnp.Array: Fisher Information Matrix of hc waveform
    """
    # Generate the waveform derivatives
    grads = grad_hc(thetas)
    # Return FIM result
    return fim_core(grads)
