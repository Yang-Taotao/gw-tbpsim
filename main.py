"""
Main script file for testing purposes.
"""

# %%
# Imports
import os
import jax
import matplotlib.pyplot as plt
import src.gw_tbpsim as gw
from src.gw_tbpsim.constant import THETA_ARRAY, F_SIG


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
# Test - grad calc
grad_hp = gw.grad_hp(THETA_ARRAY)
grad_hc = gw.grad_hc(THETA_ARRAY)


# %%
# Test - grad plot
def grad_plot(f_sig, d_hp, d_hc):
    """Simple gradiant plot display"""
    fig, ax = plt.subplots()
    ax.plot(f_sig, d_hp.real[:, 0], label="d hp.real")
    ax.plot(f_sig, d_hc.imag[:, 0], label="d hc.imag")
    ax.set(xlabel="sig", ylabel="d_h", xscale="log")
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close()


grad_plot(F_SIG, grad_hp, grad_hc)
