#!/bin/bash

SIZE=100
RIM=5
ITERATIONS=3

echo "Live presentation of the Matrix library."
echo "  June 02, 2015 -- Felix Niederwanger -- BiCGStab Solver in OpenCL"


echo "Performance tests on CPU and GPU ... "
time ./performance -n $SIZE --cpu | tee perf_cpu
time ./performance -n $SIZE --gpu | tee perf_gpu

echo "All good. Now running some random test cases .... "

./matrix_cl -n $SIZE --rim $RIM -i $ITERATIONS --cpu | tee test_cpu
./matrix_cl -n $SIZE --rim $RIM -i $ITERATIONS --gpu | tee test_gpu

