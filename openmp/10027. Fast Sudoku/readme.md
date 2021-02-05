# [10027. Fast Sudoku](https://judgegirl.csie.org/problem/0/10027)

## Concept
Do two phase of `dfs` to reach the good load balance.
- First phase: do `dfs` to certain `depth` (single thread).
- Second phase: do `dfs` from that `depth` to final (multithread).

## Compile
`$ gcc sudoku.c -fopenmp -O2`
