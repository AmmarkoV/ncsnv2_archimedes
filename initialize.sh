#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

ORIG_DIR=`pwd`


python3 -m venv ncsnv2Env
source ncsnv2Env/bin/activate
python3 -m pip install -r requirements.txt


mkdir -p exp/datasets
cd exp/datasets/
git clone https://github.com/fyu/lsun
cd lsun
python3 download.py

exit 0
