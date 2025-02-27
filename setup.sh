#!/bin/bash
echo "Creating Conda environment from environment.yaml..."
conda env create -f environment.yaml

echo "Activating the environment..."
conda activate bp

echo "Installing other dependencies..."
