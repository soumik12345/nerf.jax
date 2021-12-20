import os
import jax
from jax.tools import colab_tpu
from matplotlib import pyplot as plt


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
