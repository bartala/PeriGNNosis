#!/bin/bash

# This script sets up the Python virtual environment and installs dependencies.

# Python 3.10+ is recommended.
echo "Creating virtual environment..."
python -m venv .venv

# Activate the virtual environment.
# The following commands will run within this environment.
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete. The virtual environment is ready."
