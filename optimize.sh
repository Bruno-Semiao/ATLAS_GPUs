#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4
#SBATCH --mem=12288MB

module load cuda
module load gcc-8.5

rm /users3/ship/u22brunosemiao/ClusterInfoOptimization/src_run/*
cp /users3/ship/u22brunosemiao/ClusterInfoOptimization/src/* /users3/ship/u22brunosemiao/ClusterInfoOptimization/src_run

nvcc /users3/ship/u22brunosemiao/ClusterInfoOptimization/src_run/Intern.cu -o Intern -lcudadevrt -rdc=true --extended-lambda -std=c++17 --expt-relaxed-constexpr -O3 -allow-unsupported-compiler -lstdc++fs

if [ -e Intern ]; then
	chmod u+x Intern
	./Intern
fi