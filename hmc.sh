#!/bin/bash

hipify-perl device_api_example.cu > device_api_example.hip
number=$( ls -ltr | tail -20 | grep build | awk '{print $9}' | awk -F '_' '{print $2}' | sed -n '$p')
mkdir build_$((number+1))
CC=hipcc CXX=hipcc cmake -Bbuild_$((number+1)) .
make -C build_$((number+1))
rm device_api_example && cp build_$((number+1))/device_api_example .
sbatch curand_test.slurm
