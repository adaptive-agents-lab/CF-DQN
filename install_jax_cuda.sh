#!/bin/bash

# Install JAX with CUDA 12 support
uv pip install "jax[cuda12]>=0.6.2,<0.7.2"

# Force CuDNN to match jaxlib (PyTorch may have pinned an older version)
uv pip install --force-reinstall --no-deps "nvidia-cudnn-cu12>=9.8"

uv run python -c "import jax; print(jax.devices())"
