



#!/bin/bash
set -e

# 1. Path to your project in scratch
cd "$(dirname "$0")"

# 2. Setup Jupyter runtime inside your scratch folder
mkdir -p .jupyter_runtime
export JUPYTER_RUNTIME_DIR=$PWD/.jupyter_runtime

echo "Set JUPYTER_RUNTIME_DIR to $JUPYTER_RUNTIME_DIR" >&2

# 3. Cache settings (HuggingFace/Torch often fill up home, so keep them in scratch)
export HF_CACHE_DIR="/home/neuronet_team288/NNti_Project/.cache"
export HF_HOME="${HF_CACHE_DIR}/huggingface"
export TORCH_HOME="${HF_CACHE_DIR}/torch"
export TORCHINDUCTOR_CACHE_DIR="${HF_CACHE_DIR}/torch_inductor"
mkdir -p $TORCHINDUCTOR_CACHE_DIR
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# 4. Launch Jupyter
# Since your Dockerfile updated the 'base' env, we call it from /opt/conda/bin/
JUPYTER_PATH=$(which jupyter-lab)
$JUPYTER_PATH "$@"
