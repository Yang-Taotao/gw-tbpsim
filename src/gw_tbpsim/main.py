'''
Test
'''
# Imports
import os
import jax
import jax.numpy as jnp

# JAX settings
# Float64 support option
jax.config.update("jax_enable_x64", False)
# Mem allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Test
print(jax.default_backend())
print(jax.devices())
print(jnp.arange(3))
