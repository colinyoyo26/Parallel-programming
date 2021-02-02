# [10107. Sparse Matrix Multiplication (CUDA)](https://judgegirl.csie.org/problem/0/10107)

It's a sparse-sparse matrix multiplication.

## Compile

`$ nvcc -Xcompiler "-O2 -fopenmp" spmm.cu -o spmm`

## Future Work
Now each block process a row of result, which seems not a good practice for load balance (maybe let each block to process a tile could make it better)

It still has a lot of space to optimize.