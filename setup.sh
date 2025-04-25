#!/bin/bash
echo "Creating Conda environment"
conda create -n slmner python=3.12

echo "Activating Conda environment"
conda activate slmner

echo "Installing dependencies..."
pip install -r src/torch_requirements.txt
pip install -r src/requirements.txt