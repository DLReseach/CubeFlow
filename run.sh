#!/usr/bin/env bash
source .envrc
pip install -r requirements.txt
python mains/cnn.py -c configs/cnn.json
