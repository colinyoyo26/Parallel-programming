# [10109. Sorting (CUDA)](https://judgegirl.csie.org/problem/0/10109)

1. Even it can pass the judge, the performace is really slow.
2. It run faster without cache the subarray to shared memory.

## Concept
[Bitonic sort](https://en.wikipedia.org/wiki/Bitonic_sorter)

## Compile
`$ nvcc -Xcompiler "-O2 -fopenmp" sort.cu -o sort`
