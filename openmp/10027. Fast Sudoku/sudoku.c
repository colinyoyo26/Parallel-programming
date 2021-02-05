#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#define unlikely(x) __builtin_expect(!!(x), 0)
#define BLKIDX(r, c) ((r / 3) * 3 + c / 3)
#define FLIP(chk, bit, r, c, b) \
    do {                        \
        chk->row[r] ^= bit;     \
        chk->col[c] ^= bit;     \
        chk->blk[b] ^= bit;     \
    } while (0)

typedef struct {
    int row[9], col[9], blk[9];
} checker_t;
typedef struct {
    int r, c;
} pos_t;
static int mask = 0x1ff, zcnt = 0;
static pos_t zpos[81];
static checker_t chkque[1 << 20];
static int qend = 0;
void search_dep(int idx, checker_t *chk, int dep)
{
    if (unlikely(idx == dep)) {
        chkque[qend++] = *chk;
        return;
    }
    int r = zpos[idx].r, c = zpos[idx].c, b = BLKIDX(r, c);
    int valid = ~(chk->row[r] | chk->col[c] | chk->blk[b]) & mask;
    for (int bit; valid; valid ^= bit) {
        bit = valid & -valid;
        FLIP(chk, bit, r, c, b);
        search_dep(idx + 1, chk, dep);
        FLIP(chk, bit, r, c, b);
    }
}

int dfs(int idx, checker_t *chk)
{
    if (unlikely(idx == zcnt))
        return 1;
    int cnt = 0;
    int r = zpos[idx].r, c = zpos[idx].c, b = BLKIDX(r, c);
    int valid = ~(chk->row[r] | chk->col[c] | chk->blk[b]) & mask;
    for (int bit; valid; valid ^= bit) {
        bit = valid & -valid;
        FLIP(chk, bit, r, c, b);
        cnt += dfs(idx + 1, chk);
        FLIP(chk, bit, r, c, b);
    }
    return cnt;
}

int main()
{
    checker_t chk;
    memset(&chk, 0, sizeof(checker_t));
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            int num, b = BLKIDX(i, j);
            scanf("%d", &num);
            int bit = (1 << num) >> 1;
            FLIP((&chk), bit, i, j, b);
            zpos[zcnt] = (pos_t){i, j};
            zcnt += !bit;
        }
    }
    int dep = zcnt >> 2, cnt = 0;
    search_dep(0, &chk, dep);
#pragma omp parallel for schedule(guided) reduction(+ : cnt)
    for (int i = 0; i < qend; i++)
        cnt += dfs(dep, &chkque[i]);
    printf("%d\n", cnt);
    return 0;
}
