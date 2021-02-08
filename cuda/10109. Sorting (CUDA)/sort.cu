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

__global__ void _bitonic_internal(uint32_t *A, int seqsz)
{
#define CACHESZ 1024
    int idx = threadIdx.x, sid = blockIdx.x, blksz = blockDim.x;
    uint32_t *ptr = &A[sid * seqsz];
    __shared__ uint32_t cache[CACHESZ];
    if (seqsz <= CACHESZ) {
        for (int i = idx; i < seqsz; i += blksz)
            cache[i] = ptr[i];
        ptr = cache;
        __syncthreads();
    }
    for (int off = seqsz >> 1; off; off >>= 1) {
        for (int i = idx; i < (seqsz >> 1); i += blksz) {
            int j = ((i & ~(off - 1)) << 1) | (i & (off - 1));
            cond_swap(&ptr[j], &ptr[j | off], sid & 1);
        }
        __syncthreads();
    }
    if (seqsz <= CACHESZ) {
        ptr = &A[sid * seqsz];
        for (int i = idx; i < seqsz; i += blksz)
            ptr[i] = cache[i];
    }
#undef CACHESZ
}

static inline void bitonic_sort(uint32_t *A, int n)
{
    int blksz = 1024;
    for (int seqsz = 2; seqsz <= n; seqsz <<= 1) {
        int t = seqsz >> 1;
        _bitonic_internal<<<n / seqsz, t> blksz ? blksz : t>> > (A, seqsz);
    }
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

static uint32_t A[MAXN], B[MAXN];
#include <algorithm>
int main()
{
    int n, key;
    uint32_t *cuA;
    cudaMalloc(&cuA, MAXN * sizeof(uint32_t));
    while (scanf("%d %d", &n, &key) == 2) {
        assert((n & -n) == n);

#ifdef DEBUG
        for (uint32_t i = 0; i < n; i++)
            printf("%u ", (i * i + key) % key);
        printf("\n\n");
#endif
        const int blksz = 512, gridsz = divceil(n, blksz);
        init_array<<<gridsz, blksz>>>(cuA, n, key);
        bitonic_sort(cuA, n);
#ifdef DEBUG
        printf("\n\n");
        cudaMemcpy(A, cuA, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++)
            printf("%u ", A[i]);
        printf("\n");
#endif
#ifdef ASSERT
        cudaMemcpy(A, cuA, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < n; i++)
            B[i] = (i * i + key) % key;
        std::sort(B, B + n);
        for (int i = 0; i < n; i++)
            assert(B[i] == A[i] || 1 == printf("%u %u %u\n", i, B[i], A[i]));
#endif
        fast_hash(cuA, n);
        print_result<<<1, 1>>>(cuA);
    }
    cudaFree(cuA);
    return 0;
}
