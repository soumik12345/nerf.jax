import jax
import optax
from jax import lax
import jax.numpy as jnp

from flax import jax_utils
from flax.training import train_state, common_utils

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model import NeRF
from .utils import init_tpu
from .ray_tracing import generate_rays, render_rays


class NeRFSystem:
    def __init__(self, image_height: int, image_width: int, focal: float) -> None:
        init_tpu()
        self.image_height = image_height
        self.image_width = image_width
        self.focal = focal
        self.n_devices = jax.local_device_count()
        self.key, self.random_number_generator = jax.random.split(jax.random.PRNGKey(0))
        self.model = NeRF()
        self.parameters = jax.jit(self.model.init)(
            {"params": self.key},
            jnp.ones((image_height * image_width, 3)),
        )["params"]
    
    @staticmethod
    def train_step(state, batch, random_number_generator):
        inputs, targets = batch

        def loss_fn(params):
            model_fn = lambda x: state.apply_fn({"params": params}, x)
            rgb, _, _ = render_rays(
                model_fn, inputs, random_number_generator=random_number_generator
            )
            return jnp.mean((rgb - targets) ** 2)

        grads = jax.grad(loss_fn)(state.params)
        grads = lax.pmean(grads, axis_name="batch")
        new_state = state.apply_gradients(grads=grads)
        return new_state
    
    @staticmethod
    @jax.jit
    def evaluate(state, test_image, test_rays):
        model_fn = lambda x: state.apply_fn({"params": state.params}, x)
        rgb, *_ = render_rays(model_fn, test_rays)
        loss = jnp.mean((rgb - test_image) ** 2)
        psnr = -10.0 * jnp.log(loss) / jnp.log(10.0)
        return rgb, psnr

    def compile(self, learning_rate: float, train_poses, test_pose):
        self.psnr_history = []
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.parameters,
            tx=optax.adam(learning_rate=learning_rate),
        )
        self.state = jax.device_put_replicated(self.state, jax.local_devices())
        self.parallel_train_step = jax.pmap(self.train_step, axis_name="batch")
        self.train_rays = np.stack(
            list(
                map(
                    lambda x: generate_rays(
                        self.image_height, self.image_width, self.focal, x
                    ),
                    train_poses,
                )
            )
        )
        self.test_rays = generate_rays(
            self.image_height, self.image_width, self.focal, test_pose
        )

    def train(self, train_images, test_image, num_iterations: int, plot_interval: int):
        for step in tqdm(range(num_iterations + 1)):
            rng_idx, rng_step = jax.random.split(
                jax.random.fold_in(self.random_number_generator, step)
            )
            sharded_random_number_generators = common_utils.shard_prng_key(rng_step)
            idx = jax.random.randint(
                rng_idx, (self.n_devices,), minval=0, maxval=len(self.train_rays)
            )
            batch = self.train_rays[tuple(idx), ...], train_images[tuple(idx), ...]
            self.state = self.parallel_train_step(
                self.state, batch, sharded_random_number_generators
            )
            if step % plot_interval == 0:
                evaluation_state = jax_utils.unreplicate(self.state)
                rgb, psnr = self.evaluate(evaluation_state, test_image, self.test_rays)
                self.psnr_history.append(np.asarray(psnr))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.imshow(rgb)
                ax1.axis("off")
                ax2.plot(np.arange(0, step + 1, plot_interval), self.psnr_history)
                plt.show()
        self.state = jax_utils.unreplicate(self.state)
