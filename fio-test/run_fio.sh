#!/bin/bash

threads=("4" "8" "12" "16")

for item in  "${threads[@]}"; do
	mkdir -p "${item}_threads"
	for i in {1..5}; do
		sudo fio --output-format=json --output="${item}_threads/r${i}.json" rread.fio --numjobs=${item}
	done
done
