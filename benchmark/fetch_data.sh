#!/bin/bash
# only download 100M points
sudo apt install git-lfs -y
git lfs install

GIT_LFS_SKIP_SMUDGE=1 git clone <https://github.com/microsoft/SPTAG.git>
cd SPTAG
git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_1.bin"
git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_2.bin"
git lfs pull --include "datasets/SPACEV1B/vectors.bin/vectors_3.bin"

git lfs pull --include "datasets/SPACEV1B/query.bin"
git lfs pull --include "datasets/SPACEV1B/truth.bin"

cd datasets/SPACEV1B/vectors.bin/
mv vectors_1.bin vectors_merged.bin
for i in {2..3}; do
	cat vectors_$i.bin >> vectors_merged.bin
	rm vectors_$i.bin
done

