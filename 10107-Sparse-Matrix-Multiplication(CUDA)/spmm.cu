#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define MAXN (int) 1e6
#define BLKSZ 32

typedef struct {
    int *rowptr, *colidx, *val;
} csr_t;

typedef struct {
    int *colptr, *rowidx, *val;
} csc_t;

typedef struct {
    int row, col, val;
} coo_t;

static __device__ __forceinline__ uint32_t rotate_left(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32 - n));
}
static __device__ __forceinline__ uint32_t encrypt(uint32_t m, uint32_t key)
{
    return (rotate_left(m, key & 31) + key) ^ key;
}

static __global__ void spmm(csr_t A, csc_t B, int BN, uint32_t *ret)
{
#define CACHESZ 512
    int tid = threadIdx.x, blksz = blockDim.x;
    int bid = blockIdx.x;
    __shared__ uint32_t colidx[CACHESZ], val[CACHESZ];
    uint32_t start = A.rowptr[bid], Annz = A.rowptr[bid + 1] - start;
    for (int i = tid; i < Annz; i += blksz) {
        colidx[i] = A.colidx[i + start];
        val[i] = A.val[i + start];
    }
    __syncthreads();

    uint32_t sum = 0;
    for (int c = tid; c < BN; c += blksz) {
        uint32_t tmp = 0;
        uint32_t l = B.colptr[c], r = B.colptr[c + 1];
        for (int i = 0; i < Annz; i++) {
            uint32_t cidx = colidx[i];
            for (; l < r && B.rowidx[l] <= cidx; l++)
                tmp += B.rowidx[l] == cidx ? val[i] * B.val[l] : 0;
        }
        sum += tmp ? encrypt((c + 1) * (bid + 1), tmp) : 0;
    }
    ret[bid * blksz + tid] = sum;
#undef CACHESZ
}

static __global__ void _add_internal(uint32_t *src, int n)
{
    int tid = threadIdx.x, blksz = blockDim.x,
        idx = blockIdx.x * (blksz << 1) + tid;
    __shared__ uint32_t cache[1024];
    cache[tid] = idx < n ? src[idx] : 0;
    cache[tid + blksz] = idx + blksz < n ? src[idx + blksz] : 0;
    for (int i = blksz; tid < i; i >>= 1) {
        __syncthreads();
        cache[tid] += cache[tid + i];
    }
    if (!tid)
        src[blockIdx.x] = cache[0];
}

static void fast_add(int N, uint32_t *src)
{
#define divceil(a, b) (a + b - 1) / b
    const int blksz = 512;
    for (int rem = N; rem != 1; rem = divceil(rem, (blksz << 1)))
        _add_internal<<<dim3(divceil(rem, (blksz << 1))), dim3(blksz)>>>(src,
                                                                         rem);
#undef divceil
}

static __global__ void print_result(uint32_t *result)
{
    printf("%u\n", *result);
}

static uint32_t Arowptr[MAXN + 1], Acolidx[MAXN], Aval[MAXN];
static uint32_t Bcolptr[MAXN + 1], Browidx[MAXN], Bval[MAXN];
static coo_t buf[MAXN];
static void init_all(int *N,
                     int *M,
                     int *R,
                     csr_t *cuA,
                     csc_t *cuB,
                     uint32_t **curet)
{
    int nA, nB;
    scanf("%d %d %d", N, M, R);
    scanf("%d %d", &nA, &nB);
    int x;
    int prev = -1;
    for (int i = 0; i < nA; i++) {
        scanf("%d %d %d", &x, &Acolidx[i], &Aval[i]);
        if (prev != x) {
            for (int j = prev + 1; j <= x; j++)
                Arowptr[j] = i;
            prev = x;
        }
    }
    for (int i = prev + 1; i <= *N; i++)
        Arowptr[i] = nA;

    for (int i = 0; i < nB; i++)
        scanf("%d %d %d", &buf[i].row, &buf[i].col, &buf[i].val);
    std::stable_sort(buf, buf + nB,
                     [](coo_t a, coo_t b) { return a.col < b.col; });
    prev = -1;
    for (int i = 0; i < nB; i++) {
        Browidx[i] = buf[i].row;
        Bval[i] = buf[i].val;
        if (prev != buf[i].col) {
            for (int j = prev + 1; j <= buf[i].col; j++)
                Bcolptr[j] = i;
            prev = buf[i].col;
        }
    }
    for (int i = prev + 1; i <= *R; i++)
        Bcolptr[i] = nB;
    cudaMalloc(&cuA->rowptr, (*N + 1) * sizeof(uint32_t));
    cudaMalloc(&cuA->colidx, nA * sizeof(uint32_t));
    cudaMalloc(&cuA->val, nA * sizeof(uint32_t));
    cudaMalloc(&cuB->colptr, (*R + 1) * sizeof(uint32_t));
    cudaMalloc(&cuB->rowidx, nB * sizeof(uint32_t));
    cudaMalloc(&cuB->val, nB * sizeof(uint32_t));
    cudaMalloc(curet, *N * BLKSZ * sizeof(uint32_t));
    cudaMemcpy(cuA->rowptr, Arowptr, (*N + 1) * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuA->colidx, Acolidx, nA * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuA->val, Aval, nA * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cuB->colptr, Bcolptr, (*R + 1) * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuB->rowidx, Browidx, nB * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuB->val, Bval, nB * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

int main()
{
    int N, M, R;
    uint32_t *curet;
    csr_t cuA;
    csc_t cuB;

    init_all(&N, &M, &R, &cuA, &cuB, &curet);
    spmm<<<dim3(N), dim3(BLKSZ)>>>(cuA, cuB, R, curet);
    fast_add(N * BLKSZ, curet);
    print_result<<<dim3(1), dim3(1)>>>(curet);
    cudaFree(cuA.rowptr);
    cudaFree(cuA.colidx);
    cudaFree(cuA.val);
    cudaFree(cuB.colptr);
    cudaFree(cuB.rowidx);
    cudaFree(cuB.val);
    cudaFree(curet);
    return 0;
}
