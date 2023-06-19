#!/usr/bin/env bash

# Assumes a conda environment with Python 3.8

# exit when any command fails
set -e
WORK_DIR="$(dirname "$(realpath -s "$0")")"

echo "WORK_DIR: WORK_DIR"
cd $ {WORK_DIR}
pip install torch torchvision
pip install timm
pip install imgaug
pip install p_tqdm
pip install pytorch-metric-learning[with-hooks-cpu]
pip install pytorch-lightning

# additional installations for CLIP
pip install git+https://github.com/openai/CLIP.git
pip install ftfy regex