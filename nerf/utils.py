import os
import wget
import numpy as np
from matplotlib import pyplot as plt

import jax
from jax.tools import colab_tpu


def init_tpu():
    if "COLAB_TPU_ADDR" in os.environ:
        colab_tpu.setup_tpu()
    print(jax.local_devices())


def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        _ = plt.imshow(images[i])
        plt.axis("off")
    plt.show()


def load_lego_data():
    if not os.path.isfile("tiny_nerf_data.npz"):
        wget.download(
            "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
        )
    data = np.load("tiny_nerf_data.npz")
    images = data["images"]
    poses = data["poses"]
    focal = float(data["focal"])
    _, image_height, image_width, _ = images.shape
    test_image, test_pose = images[101], poses[101]
    train_images = images[:100, ..., :3]
    train_poses = poses[:100]
    return (
        train_images,
        train_poses,
        test_image,
        test_pose,
        focal,
        image_height,
        image_width,
    )
