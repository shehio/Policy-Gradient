#!/bin/bash
brew install swig
brew install ffmpeg

# Create virtual environment
python3.13 -m venv venv
source ./venv/bin/activate
which python
which pip
pip install -r requirements.txt
AutoROM --accept-license
pre-commit install
