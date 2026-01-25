#!/bin/bash
set -e

# IMPORTANT: Set this to your MinIO node's IP address
MINIO_HOST="<MINIO_NODE_IP>"

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
wget https://github.com/milvus-io/milvus/releases/download/v2.3.21/milvus-standalone-docker-compose.yml \
  -O docker-compose.yml

# Modify docker-compose.yml to use external MinIO
cat > docker-compose.yml <<EOF
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.21
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: ${MINIO_HOST}:9000
      MINIO_ACCESS_KEY_ID: minio
      MINIO_SECRET_ACCESS_KEY: minio123
      MINIO_USE_SSL: false
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"

networks:
  default:
    name: milvus
EOF

# Start Milvus
export MINIO_HOST
sudo -E docker-compose up -d
