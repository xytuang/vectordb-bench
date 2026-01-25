#!/bin/bash
set -e

# Set limiton number of open files
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

# Milvus standalone
wget https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml \
  -O docker-compose.yml

rm docker-compose.yml
cp mod-docker-compose.yml docker-compose.yml
sudo docker-compose up -d

wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/
mc alias set remote http://node0:9000 minio minio123
mc ls remote

