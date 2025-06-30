#!/bin/bash

# Ensure the script is run as bash script.sh, not sh script.sh
if [ -z "$BASH_VERSION" ]; then
    echo "Please run this script using bash, not sh" >&2
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda could not be found, please install Anaconda or Miniconda first."
    exit 1
fi

# Create UnityLogs directory
echo "Creating UnityLogs directory..."
mkdir -p ../UnityLogs

# Create a conda environment with Python 3.10.11
echo "Creating conda environment 'crew_old' with Python 3.10.11..."
conda create -n crew_old python=3.10.11 -y

# # Activate the environment and install PyAudio
echo "Activating the 'crew_old' environment and installing PyAudio..."
# conda init
source ~/Desktop/installs/miniconda3/etc/profile.d/conda.sh
conda activate crew_old
conda install -c conda-forge pyaudio -y

# Check if Poetry is installed

curl -sSL https://install.python-poetry.org | python3 -
# Configure PATH in .bashrc for Poetry
echo "export PATH=\"/home/$USER/.local/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc


# Install project dependencies using Poetry
echo "Installing dependencies using Poetry..."
poetry install

echo "Setup completed successfully."
