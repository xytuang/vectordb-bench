#!/bin/bash

# Install go
wget https://go.dev/dl/go1.25.6.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.25.6.linux-amd64.tar.gz
echo "export PATH=$PATH:/usr/local/go/bin" >> $HOME/.profile
source $HOME/.profile

# Install and start docker
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

mkdir -p /mydata/minio-data

sudo docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v /mydata/minio-data:/data \
  -e MINIO_ROOT_USER=minio \
  -e MINIO_ROOT_PASSWORD=minio123 \
  minio/minio server /data --console-address ":9001"

wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
sudo mv mc /usr/local/bin/

mc alias set local http://localhost:9000 minio minio123

mc mb local/milvus









