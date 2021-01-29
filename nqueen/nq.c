#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
 
typedef struct {
    int col, rdiag, ldiag;
} checker_t;
 
static int obs[20], nmask;
 
static inline int valid_col(const checker_t chk, const int r)
{
    return ~(chk.col | chk.rdiag | chk.ldiag | obs[r]) & nmask;
}
static inline checker_t next_checker(const checker_t chk,
                                     const int r,
                                     const int cpos)
{
    return (checker_t){.col = chk.col | cpos,
                       .rdiag = (chk.rdiag | cpos) << 1,
                       .ldiag = (chk.ldiag | cpos) >> 1};
}
 
static checker_t chkque[524288];
static int qend;
int dfs(const int r, const checker_t chk, const int n)
{
    int cmask = valid_col(chk, r), cnt = 0;
    if (r == n - 1) {
        if (nmask != (1 << n) - 1)
            for (int m = cmask; m; m ^= m & -m)
                chkque[qend++] = next_checker(chk, r, m & -m);
        return cmask ? __builtin_popcount(cmask) : 0;
    }
    for (; cmask; cmask ^= cmask & -cmask)
        cnt += dfs(r + 1, next_checker(chk, r, cmask & -cmask), n);
    return cnt;
}
 
static inline int cnt_nqueens(const int n, const checker_t chk)
{
    int cnt = 0, tmp = n, search_dep = n / 3 - 1, nsol;
    search_dep = search_dep < 1 ? 1 : search_dep;
    nmask = (1 << n) - 1;
    qend = 0;
    nsol = dfs(0, chk, search_dep);
#pragma omp parallel for reduction(+ : cnt) schedule(dynamic)
    for (int i = 0; i < nsol; i++)
        cnt += dfs(search_dep, chkque[i], n);
    return cnt;
}
 
int main()
{
    for (int casecnt = 1, n; scanf("%d", &n) == 1; casecnt++) {
        checker_t chk = {0, 0, 0};
        for (int i = 0; i < n; i++) {
            char buf[20], *ptr = buf - 1;
            scanf("%s", buf);
            obs[i] = 0;
            while ((ptr = strchr(ptr + 1, '*'))) {
                int shft = ((intptr_t) ptr - (intptr_t) buf) / sizeof(char);
                obs[i] |= 1 << shft;
            }
        }
        printf("Case %d: %d\n", casecnt, cnt_nqueens(n, chk));
    }
    return 0;
}
