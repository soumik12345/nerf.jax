import jax
from jax import lax
import jax.numpy as jnp

import flax.linen as nn

from typing import Any, Callable, Sequence


class NeRF(nn.module.Module):

    num_layers: int = 8
    num_units: int = 256
    skips: Sequence[int] = (4,)
    periodic_functions: Sequence[Callable] = (jnp.sin, jnp.cos)
    out_channels: int = 4
    use_embedding: bool = True
    l_embed: int = 6
    dtype: Any = jnp.float32
    precision: Any = lax.Precision.DEFAULT

    def embed(self, inputs):
        batch_size, _ = inputs.shape
        inputs_freq = jax.vmap(lambda x: inputs * 2.0 ** x)(jnp.arange(self.l_embed))
        functions = jnp.stack([fn(inputs_freq) for fn in self.periodic_functions])
        functions = functions.swapaxes(0, 2).reshape([batch_size, -1])
        functions = jnp.concatenate([inputs, functions], axis=-1)
        return functions
    
    @nn.compact
    def __call__(self, inputs_points):
        x = self.embed(inputs_points) if self.use_embedding else inputs_points
        for i in range(self.num_layers):
            x = nn.Dense(
                self.num_units,
                dtype=self.dtype,
                precision=self.precision
            )(x)
            x = nn.relu(x)
            if i in self.skips:
                x = jnp.concatenate([x, inputs_points], axis=-1)
        return nn.Dense(
            self.out_channels,
            dtype=self.dtype,
            precision=self.precision
        )(x)
