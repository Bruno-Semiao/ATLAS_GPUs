#!/bin/bash

nvcc Intern.cu -o Intern -lcudadevrt -rdc=true --extended-lambda -std=c++17 --expt-relaxed-constexpr -O3 -allow-unsupported-compiler -lstdc++fs

#./Intern