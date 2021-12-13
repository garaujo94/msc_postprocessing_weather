#!/bin/bash

echo "Installing PyTorch..."
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "Installing DGL..."
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html

echo "OK"