#!/bin/bash
sudo apt update
sudo apt install -y python3-pip python3-venv
python3 -m venv vectordb_bench
source vectordb_bench/bin/activate
pip3 install pymilvus boto3 numpy pandas pyarrow


