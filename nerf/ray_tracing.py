import jax
import numpy as np
from jax import lax
import jax.numpy as jnp

import flax
import flax.linen as nn


def generate_rays(image_height, image_width, focal, pose):
    i, j = np.meshgrid(np.arange(image_width), np.arange(image_height), indexing="xy")
    k = -np.ones_like(i)
    i = (i - image_width * 0.5) / focal
    j = -(j - image_height * 0.5) / focal
    directions = np.stack([i, j, k], axis=-1)
    camera_matrix = pose[:3, :3]
    ray_directions = np.einsum("ijl,kl", directions, camera_matrix)
    ray_origins = np.broadcast_to(pose[:3, -1], ray_directions.shape)
    return np.stack([ray_origins, ray_directions])


def render_rays(
    model_fn,
    rays,
    near_bound=2.0,
    far_bound=6.0,
    num_samples=64,
    batch_size=10000,
    random_number_generator=None,
):
    ray_origins, ray_directions = rays
    z_vals = np.linspace(near_bound, far_bound, num_samples)
    z_shape = ray_origins.shape[:-1] + (num_samples,)
    if random_number_generator is not None:
        z_vals += (
            jax.random.uniform(random_number_generator, z_shape)
            * (far_bound - near_bound)
            / num_samples
        )
    points = (
        ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
    )
    raw = lax.map(model_fn, jnp.reshape(points, [-1, batch_size, 3]))
    raw = jnp.reshape(raw, points.shape[:-1] + (4,))
    sigma_a = nn.relu(raw[..., 3])
    rgb = nn.sigmoid(raw[..., :3])
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = jnp.concatenate(
        [dists, np.broadcast_to([1e10], dists[..., :1].shape)], axis=-1
    )
    alpha = 1.0 - jnp.exp(-sigma_a * dists)
    alpha_ = jnp.clip(1.0 - alpha, 1e-10, 1.0)
    trans = jnp.concatenate([jnp.ones_like(alpha_[..., :1]), alpha_[..., :-1]], -1)
    weights = alpha * jnp.cumprod(trans, -1)
    rgb_map = jnp.einsum("...k,...kl", weights, rgb)
    depth_map = jnp.einsum("...k,...k", weights, z_vals)
    acc_map = jnp.einsum("...k->...", weights)
    return rgb_map, depth_map, acc_map
