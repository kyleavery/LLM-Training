#!/bin/bash

# install uv (and fix ~/.config on prime boxes)
sudo chmod -R 777 ~/.config
curl -LsSf https://astral.sh/uv/install.sh | sh

# install RLVR training requirements
/home/ubuntu/.local/bin/uv venv
source .venv/bin/activate
/home/ubuntu/.local/bin/uv pip install --no-deps -r requirements_rlvr.txt

# install cuda toolkit
sudo apt update
sudo apt install -y cuda-toolkit || sudo apt install -y cuda-toolkit --fix-missing
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
