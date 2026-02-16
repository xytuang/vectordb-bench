#!/bin/bash
set -e  # Exit on error

echo "Setting up SPACEV1B dataset (100M vectors)..."

# Create data directory
sudo mkdir -p /mydata/spacev1b
sudo chown $USER:e6897-PG0 /mydata/spacev1b
cd /mydata/spacev1b

# Install git-lfs
sudo apt update
sudo apt install -y git-lfs
git lfs install

# Clone repo without downloading large files
echo "Cloning SPTAG repository..."
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/microsoft/SPTAG.git
cd SPTAG

# Download only the files we need (first 3 vector files = ~100M vectors)
echo "Downloading vector files (this will take a while)..."
git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_1.bin"
git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_2.bin"
git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_3.bin"

echo "Downloading query and ground truth files..."
git lfs pull --include "datasets/SPACEV1B/query.bin"
git lfs pull --include "datasets/SPACEV1B/truth.bin"


# Move to cleaner location
echo "Organizing files..."
cd /mydata/spacev1b
mv SPTAG/datasets/SPACEV1B/vectors.bin/vectors_1.bin ./spacev1b_vectors_1.bin 2>/dev/null || true
mv SPTAG/datasets/SPACEV1B/vectors.bin/vectors_2.bin ./spacev1b_vectors_2.bin 2>/dev/null || true
mv SPTAG/datasets/SPACEV1B/vectors.bin/vectors_3.bin ./spacev1b_vectors_3.bin 2>/dev/null || true
mv SPTAG/datasets/SPACEV1B/query.bin ./spacev1b_query.bin 2>/dev/null || true
mv SPTAG/datasets/SPACEV1B/truth.bin ./spacev1b_truth.bin 2>/dev/null || true

# Clean up git repo to save space
echo "Cleaning up to save space..."
rm -rf SPTAG

# Verify files and show space usage
echo ""
echo "=== Download Complete ==="
ls -lh spacev1b_*.bin
echo ""
echo "Disk usage:"
df -h /mydata
echo ""
