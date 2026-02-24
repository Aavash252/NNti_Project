#!/usr/bin/env bash
set -euo pipefail
export USER=condor
export LOGNAME=condor
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${UID} 
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
python /app/train_model.py
