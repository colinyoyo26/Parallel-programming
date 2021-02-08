#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#define MAXN 16777216
#define divceil(a, b) (a + b - 1) / b

__device__ __forceinline__ uint32_t encrypt(uint32_t m, uint32_t key)
{
    return (m * m + key) % key;
}

__global__ void init_array(uint32_t *A, int n, int key)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[idx] = encrypt(idx, key);
}

__device__ __forceinline__ void cond_swap(uint32_t *a, uint32_t *b, bool cond)
{
    if ((*a < *b) == cond && *a != *b) {
        uint32_t tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

__global__ void _bitonic_internal(uint32_t *A, int lgseqsz, int off)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = ((tid & ~(off - 1)) << 1) | (tid & (off - 1));
    int sid = tid >> (lgseqsz - 1);
    cond_swap(&A[idx], &A[idx | off], sid & 1);
}

static inline void bitonic_sort(uint32_t *A, int n)
{
    int dblksz = n > 1024 ? 1024 : n;
    int lgn = __builtin_ctz(n);
    for (int lgseqsz = 1; lgseqsz <= lgn; lgseqsz++)
        for (int off = 1 << (lgseqsz - 1); off; off >>= 1)
            _bitonic_internal<<<(n / dblksz), (dblksz >> 1)>>>(A, lgseqsz, off);
}

__global__ void _add_internal(uint32_t *A, int n)
{
    int tid = threadIdx.x, blksz = blockDim.x,
        idx = blockIdx.x * (blksz << 1) + tid;
    __shared__ uint32_t cache[1024];
    cache[tid] = idx < n ? A[idx] : 0;
    cache[tid + blksz] = idx + blksz < n ? A[idx + blksz] : 0;
    for (int i = blksz; i && tid < i; i >>= 1) {
        __syncthreads();
        cache[tid] += cache[tid + i];
    }
    if (!tid)
        A[blockIdx.x] = cache[0];
}

__global__ void _hash_init(uint32_t *A, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[idx] *= idx;
}

static inline void fast_hash(uint32_t *A, int n)
{
    const int blksz = 512;
    _hash_init<<<divceil(n, blksz), blksz>>>(A, n);
    for (int rem = n; rem > 1; rem = divceil(rem, (blksz << 1)))
        _add_internal<<<dim3(divceil(rem, (blksz << 1))), dim3(blksz)>>>(A,
                                                                         rem);
}

__global__ void print_result(uint32_t *A)
{
    printf("%u\n", *A);
}

int main()
{
    int n, key;
    uint32_t *cuA;
    cudaMalloc(&cuA, MAXN * sizeof(uint32_t));
    while (scanf("%d %d", &n, &key) == 2) {
        assert((n & -n) == n);
        const int blksz = 512, gridsz = divceil(n, blksz);
        init_array<<<gridsz, blksz>>>(cuA, n, key);
        bitonic_sort(cuA, n);
        fast_hash(cuA, n);
        print_result<<<1, 1>>>(cuA);
    }
    cudaFree(cuA);
    return 0;
}
