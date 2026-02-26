#!/bin/bash
set -e

# Set limit on number of open files
ulimit -n 1048576

# Set max number of vmas
sudo sysctl -w vm.max_map_count=262144

# Set max number of connections
sudo sysctl -w net.core.somaxconn=4096

# CPU performance mode
sudo apt update
sudo apt install -y linux-tools-common linux-tools-$(uname -r)
sudo cpupower frequency-set -g performance || true

# Install docker
sudo apt install -y docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker

# Download base Milvus configuration
wget https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml \
  -O docker-compose.yml

wget https://raw.githubusercontent.com/milvus-io/milvus/v2.6.11/configs/milvus.yaml

mkdir -p /mydata/vectordb-bench/benchmark_spacev1b/milvus/volumes/milvus
echo "Fetched docker compose, you should edit it to optimize for storage. See https://milvus.io/docs/diskann.md#Index-building-params"
echo "Change docker-compose.yml such that volumes for milvus-standalone points to /mydata/vectordb-bench/benchmark_spacev1b/milvus/volumes/milvus"
