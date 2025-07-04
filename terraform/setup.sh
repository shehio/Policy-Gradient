#!/bin/bash
set -e

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
fi

# Clone the repo
if [ ! -d "$REPO_NAME" ]; then
  git clone "$REPO_URL"
fi
cd "$REPO_NAME"

# Create venv if not exists
if [ ! -d "venv" ]; then
  python3.10 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt

# Download ROMs for gym (if needed)
if grep -q AutoROM requirements.txt; then
  AutoROM --accept-license
fi

# Run the main script from the scripts directory
python scripts/pgpong.py 