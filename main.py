"""
Main script file for testing purposes.
"""

# %%
# Imports
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import src.gw_tbpsim as gw
from src.gw_tbpsim.constant import THETA_ARRAY, MC_ARRAY, ETA_ARRAY

# %%
# JAX settings
jax.config.update("jax_enable_x64", True)
# Setting - Manual memory allocation -> set to true if OOM occurs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# Setting - Set up presistent cache
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_dir", ".jaxcache")

# %%
# Test - det
det_hp = gw.log_sqrt_det_hp(THETA_ARRAY)
det_hc = gw.log_sqrt_det_hc(THETA_ARRAY)
print(det_hp, det_hc)

# %%
# Test - log density param
# Recreate MC, ETA to (10,) shape
MC_ARRAY = jnp.linspace(1.0, 21.0, 10, dtype=jnp.float64)
ETA_ARRAY = jnp.linspace(0.05, 0.25, 10, dtype=jnp.float64)

# Build (8,) -> (10, 10, 8)
THETAS = jnp.tile(THETA_ARRAY, (10, 10, 1)).astype(jnp.float64)
# Replace first two entry of (i, j) with (mc, eta)
THETAS = THETAS.at[:, :, 0].set(MC_ARRAY[:, jnp.newaxis])
THETAS = THETAS.at[:, :, 1].set(ETA_ARRAY[jnp.newaxis, :])


# Vectorization
@jax.jit
def log_den_hp(thetas):
    return jax.vmap(jax.vmap(gw.log_sqrt_det_hp))(thetas)


@jax.jit
def log_den_hc(thetas):
    return jax.vmap(jax.vmap(gw.log_sqrt_det_hc))(thetas)


# %%
# Test - log density Results
l_hp = log_den_hp(THETAS)
l_hc = log_den_hc(THETAS)

# %%
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
    save_path = f".figures/log_fim_contour_{waveform}_{mc_min}_{mc_max}.png"
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


log_fim_contour(MC_ARRAY, ETA_ARRAY, l_hp)
