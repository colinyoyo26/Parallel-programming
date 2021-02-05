# [10026. Fast N-Queens](https://judgegirl.csie.org/problem/0/10026)

## Concept
Do two phase of `dfs` to reach the good load balance.
- First phase: do `dfs` to certain `depth` (single thread).
- Second phase: do `dfs` from that `depth` to final (multithread).

## Compile
`$ gcc nq.c -fopenmp -O2`
