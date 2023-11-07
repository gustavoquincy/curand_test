#!/bin/bash

number=$( ls -ltr | tail -20 | grep build | awk '{print $9}' | awk -F '_' '{print $2}' | sed -n '$p')
rm device_api_example && cp build_$((number))/device_api_example .
sbatch curand_test.slurm
