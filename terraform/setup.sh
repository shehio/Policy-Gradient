#!/bin/bash
set -e

echo "Starting setup script..."
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"

REPO_URL="https://github.com/shehio/Policy-Gradient.git"
REPO_NAME=$(basename -s .git "$REPO_URL")

# Ensure Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
  echo "Python 3.10 not found. Installing..."
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils
  # Ensure pip for python3.10
  curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.10
else
  echo "Python 3.10 found, ensuring venv package is installed..."
  sudo apt-get update
  sudo apt-get install -y python3.10-venv
fi

echo "Repository already cloned by user_data script, continuing with setup..."
echo "Repository directory: $(pwd)"

# Create necessary directories
mkdir -p models

# Create venv if not exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3.10 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate
echo "Installing requirements..."
pip install -r requirements.txt

# Download ROMs for gym (if needed)
if grep -q AutoROM requirements.txt; then
  echo "Downloading ROMs for gym..."
  AutoROM --accept-license
fi

# Run the main script from the scripts directory
echo "Setup complete! Running the main script..."
echo "Script will run in the background and continue even if it fails..."
nohup python scripts/pgpong.py > training.log 2>&1 &
echo "Training started in background. Check training.log for output."
