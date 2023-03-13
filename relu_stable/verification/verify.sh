#!/bin/bash
mkdir -p $1/verification
julia --color=yes ./verification/verify_MNIST.jl $1 $2 $3 $4
