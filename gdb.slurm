#!/bin/bash
#SBATCH -J curand_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -p xahdnormal
#SBATCH --gres=dcu:4
#SBATCH -e %j.err
#SBATCH -o %j.out

gdb ./device_api_example core.25093 
