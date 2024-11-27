"""
Core methods for gw_tbsim package.
"""

# Imports
# =========================================================================== #
import os
import jax
import jax.numpy as jnp
import tqdm
import matplotlib.pyplot as plt
from ripple.waveforms import IMRPhenomXAS

# Import constants
from .config import F_SIG, F_REF, F_PSD, F_DIFF

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
# =========================================================================== #
# FIM - Batching


def mock_compile(thetas: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Mock compilation function for running log_den_hp() and log_den_hc()
    with thetas.shape at (10, 10, 8).

    Args:
        thetas (jax.Array): GW parameter matrix at (10, 10, 8) shape.

    Returns:
        tuple[jax.Array, jax.Array]: Tuple of log density distributions
            in the form of (log_density_hp, log_density_hc)
            at ((10, 10, 8), (10, 10, 8)) shape.
    """
    # Compilation
    with tqdm.tqdm(total=2, desc="Compiling") as pbar:
        # Compile log_den_hp()
        hp_result = log_den_hp(thetas)
        pbar.update(1)
        # Compile log_den_hc()
        hc_result = log_den_hc(thetas)
        pbar.update(1)
        # Print
        print(f"Functions compiled for theta entry at {thetas.shape} shape.")
    # Func return
    return hp_result, hc_result


# Overall batch call
def log_density_batch(
    thetas: jax.Array, mcs: jax.Array, etas: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Batched opearation for log density calculation for both hp and hc waveform.

    Args:
        thetas (jax.Array): GW parameter of shape (8, )
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].
        mcs (jax.Array): Chirp mass parameter array at (n, ) shape.
        etas (jax.Array): Symmetric mass ratio parameter array at (n, ) shape.

    Returns:
        tuple[jax.Array, jax.Array]: Tuple of log density distributions
            in the form of (log_density_hp, log_density_hc)
            at ((n, n, 8), (n, n, 8)) shape.
    """
    # Local repo
    theta_dim = mcs.shape[0]
    batch_size = 10
    # Get total number of batches
    batch_num = (theta_dim + batch_size - 1) // batch_size

    # Build theta parameters
    theta = theta_gen(thetas, mcs, etas)
    # Build log density results array
    hp_result = jnp.empty((theta_dim, theta_dim))
    hc_result = jnp.empty((theta_dim, theta_dim))

    # Batching
    with tqdm.tqdm(total=batch_num**2, desc="Generating Log Density") as pbar:
        # Linear iteration over rows and columns
        for i in range(batch_num):
            for j in range(batch_num):
                # Set min, max for i, j
                i_min = i * batch_size
                j_min = j * batch_size
                i_max = min((i + 1) * batch_size, theta_dim)
                j_max = min((j + 1) * batch_size, theta_dim)

                # Slice theta into component theta_batch
                theta_batch = theta[i_min:i_max, j_min:j_max]
                # Get component log density result
                log_den_hp_batch = log_den_hp(theta_batch)
                log_den_hc_batch = log_den_hc(theta_batch)

                # Assemble log density results
                hp_result = hp_result.at[i_min:i_max, j_min:j_max].set(log_den_hp_batch)
                hc_result = hc_result.at[i_min:i_max, j_min:j_max].set(log_den_hc_batch)
                # Update pbar
                pbar.update(1)

    # Func return
    return hp_result, hc_result


# =========================================================================== #
#


@jax.jit
def log_den_hp(thetas: jax.Array) -> jax.Array:
    """
    Log density distribution of hp waveforms. Applied jax.jit().

    Args:
        thetas (jax.Array): GW parameter matrix at (n, n, 8) shape.

    Returns:
        jax.Array: Log density distribution at (n, n) shape.
    """
    # Func return
    return jax.vmap(jax.vmap(log_sqrt_det_hp))(thetas)


@jax.jit
def log_den_hc(thetas: jax.Array) -> jax.Array:
    """
    Log density distribution of hc waveforms. Applied jax.jit().

    Args:
        thetas (jax.Array): GW parameter matrix at (n, n, 8) shape.

    Returns:
        jax.Array: Log density distribution at (n, n) shape.
    """
    # Func return
    return jax.vmap(jax.vmap(log_sqrt_det_hc))(thetas)


# =========================================================================== #
# FIM - Main ==> log.sqrt.det.FIM => density statictics


def log_sqrt_det_hp(theta: jax.Array) -> jax.Array:
    """
    Calculate intermediate statistics for density building
    with log.sqrt.det.Metric on hp waveform based results.

    Args:
        theta (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jax.Array: intermediate statistics
            as log.sqrt.det.Metric.
    """
    # Calculation
    metric = projected_fim_hp(theta)
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(metric)))


def log_sqrt_det_hc(theta: jax.Array) -> jax.Array:
    """
    Calculate intermediate statistics for density building
    with log.sqrt.det.Metric on hc waveform based results.

    Args:
        theta (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jax.Array: intermediate statistics
            as log.sqrt.det.Metric.
    """
    # Calculation
    metric = projected_fim_hc(theta)
    # Func return - log density
    return jnp.log(jnp.sqrt(jnp.linalg.det(metric)))


# %%
# FIM - Projected and simple FIM


def projected_fim_hp(thetas: jax.Array) -> jax.Array:
    """
    Projected Fisher Information Matrix function call for hp waveforms.

    Args:
        thetas (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jax.Array: Projected metric on mc, eta space for hp waveforms.
    """
    # Get full FIM and dimensions
    full_fim = fim_hp(thetas)
    # Calculate the conditioned matrix for phase
    gamma = fim_phic(full_fim)
    # Calculate the conditioned matrix for time
    metric = fim_tc(gamma)
    # Func return
    return metric


def projected_fim_hc(thetas: jax.Array) -> jax.Array:
    """
    Projected Fisher Information Matrix function call for hc waveforms.

    Args:
        thetas (jax.Array): GW param
            as [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination].

    Returns:
        jax.Array: Projected metric on mc, eta space for hc waveforms.
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
        jax.Array: projected conditional matrix gamma.
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


def fim_tc(gamma: jax.Array) -> jax.Array:
    """
    Project conditional matrix back onto tc with Eq. 18 of 1311.7174.

    Args:
        gamma (jnp.ndarray): conditional matrix gamma.

    Returns:
        jax.Array: projected metric.
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


def fim_core(grads: jax.Array) -> jax.Array:
    """
    Fisher Information Matrix builder.

    Args:
        grads (jax.Array): GW waveform gradients
            with shape (F_SIG.shape[0], THETA_ARRAY.shape[-1]).

    Returns:
        jax.Array: Fisher Information Matrix.
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


def fim_hp(thetas: jax.Array) -> jax.Array:
    """
    Build Fisher Information Matrix for hp waveform.

    Args:
        thetas (jax.Array): GW param
            [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination]

    Returns:
        jax.Array: Fisher Information Matrix of hp waveform
    """
    # Generate the waveform derivatives
    grads = grad_hp(thetas)
    # Return FIM result
    return fim_core(grads)


def fim_hc(thetas: jax.Array) -> jax.Array:
    """
    Build Fisher Information Matrix for hc waveform.

    Args:
        thetas (jax.Array): GW param
            [mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination]

    Returns:
        jax.Array: Fisher Information Matrix of hc waveform
    """
    # Generate the waveform derivatives
    grads = grad_hc(thetas)
    # Return FIM result
    return fim_core(grads)


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


# =========================================================================== #
# Figs


def log_fim_contour(
    data_x: jnp.ndarray,
    data_y: jnp.ndarray,
    data_z: jnp.ndarray,
    waveform: str = "hp",
):
    """
    Generate contourf plots for log density wrt mc, eta
    Defaulted at waveform hp results
    """
    # Local plotter resources
    xlabel, ylabel, cblabel = (
        r"Chirp Mass $\mathcal{M} [M_\odot$]",
        r"Symmetric Mass Ratio $\eta$",
        r"$\log$ Template Bank Density",
    )
    mc_min, mc_max = jnp.min(data_x), jnp.max(data_x)
    save_path = f"./figures/log_fim_contour_{waveform}_{mc_min}_{mc_max}.png"
    # Plot init
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotter
    cs = ax.contourf(
        data_x,
        data_y,
        data_z.T,
        alpha=0.8,
        levels=20,
        cmap="gist_heat",
    )
    # Plot customization
    ax.set(xlabel=xlabel, ylabel=ylabel)
    cb = plt.colorbar(cs, ax=ax)
    cb.ax.set_ylabel(cblabel)
    # Plot admin
    fig.savefig(save_path)
