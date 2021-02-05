# [10084. Prefix Sum (pthread)](https://judgegirl.csie.org/problem/0/10084)

`psum.c` never join the threads, which lead to a better performance.

## Concept
Calculate the prefix sum in two phase.
- Phase 1: Calculate the prefix sum in individual block.
- Phase 2: Add the sum of last elements of previous blocks to all the element in this block.

## Compile
`$ gcc -std=c99 -O2 -pthread psum.c secret.c -o psum`
