#!/bin/bash

brew install swig

# Create virtual environment
python3.11 -m venv venv   
source ./venv/bin/activate
which python
which pip
pip install -r requirements.txt
AutoROM --accept-license