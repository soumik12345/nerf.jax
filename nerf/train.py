import jax
import optax
from jax import lax
import jax.numpy as jnp

from flax import jax_utils
from flax.training import train_state, common_utils

import numpy as np
from tqdm import tqdm
from imageio import mimwrite

from .model import NeRF
from .utils import init_tpu, plot_results
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
        self.define_matrices()

    def define_matrices(self):
        self.translation = lambda t: np.asarray(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ]
        )
        self.rotation_phi = lambda phi: np.asarray(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )
        self.rotation_theta = lambda th: np.asarray(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def train_step(state, batch, random_number_generator):
        inputs, targets = batch

        def loss_fn(params):
            model_fn = lambda x: state.apply_fn({"params": params}, x)
            rgb, _ = render_rays(
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
        rgb, depth_map = render_rays(model_fn, test_rays)
        loss = jnp.mean((rgb - test_image) ** 2)
        psnr = -10.0 * jnp.log(loss) / jnp.log(10.0)
        return rgb, depth_map, psnr

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
                rgb, depth_map, psnr = self.evaluate(
                    evaluation_state, test_image, self.test_rays
                )
                self.psnr_history.append(np.asarray(psnr))
                plot_results([rgb, depth_map], ["RGB Image", "Depth Map"])
        self.inference_state = jax_utils.unreplicate(self.state)

    def render_video(self, frame_rate: int = 30, quality: int = 7):
        def pose_spherical(theta, phi, radius):
            pose = self.translation(radius)
            pose = self.rotation_phi(phi / 180.0 * np.pi) @ pose
            pose = self.rotation_theta(theta / 180.0 * np.pi) @ pose
            return (
                np.array(
                    [
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]
                    ]
                ) @ pose
            )

        @jax.jit
        def get_frames(rays):
            model_fn = lambda x: self.inference_state.apply_fn(
                {"params": self.inference_state.params}, x
            )
            rgb_image, depth_map = render_rays(model_fn, rays)
            rgb_image = (255 * jnp.clip(rgb_image, 0, 1)).astype(jnp.uint8)
            return rgb_image, depth_map

        video_angle = jnp.linspace(0.0, 360.0, 120, endpoint=False)
        video_pose = map(lambda th: pose_spherical(th, -30.0, 4.0), video_angle)
        rays = np.stack(
            list(
                map(
                    lambda x: generate_rays(
                        self.image_height, self.image_width, self.focal, x[:3, :4]
                    ),
                    video_pose,
                )
            )
        )
        rgb_frames, depth_maps = lax.map(get_frames, rays)
        rgb_frames = map(np.asarray, rgb_frames)
        depth_maps = map(np.asarray, depth_maps)
        mimwrite("rgb.mp4", tuple(rgb_frames), fps=frame_rate, quality=quality)
        mimwrite("depth.mp4", tuple(depth_maps), fps=frame_rate, quality=quality)
