import os
import jax
from jax.tools import colab_tpu


def init_tpu():
    if "COLAB_TPU_ADDR" in os.environ:
        colab_tpu.setup_tpu()
    print(jax.local_devices())
