#include <stdint.h>
#include <stdio.h>

#define MAXN 16777216
#define TASKS 64
#define BLKSZ 512
#define GRIDSZ (N + BLKSZ * TASKS - 1) / (BLKSZ * TASKS)
static __device__ __forceinline__ uint32_t rotate_left(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32 - n));
}

static __device__ __forceinline__ uint32_t encrypt(uint32_t m, uint32_t key)
{
    return (rotate_left(m, key & 31) + key) ^ key;
}

static __global__ void dot_product(uint32_t C[],
                                   uint32_t key1,
                                   uint32_t key2,
                                   uint32_t N)
{
    int tid = threadIdx.x, idx = (blockIdx.x * blockDim.x + tid) * TASKS;
    uint32_t sum = 0;
    int bound = idx + TASKS < N ? idx + TASKS : N;
    for (int i = idx; i < bound; i++)
        sum += encrypt(i, key1) * encrypt(i, key2);
    C[blockIdx.x * blockDim.x + tid] = sum;
}

static __global__ void _add_internal(uint32_t *src, uint32_t N)
{
    int tid = threadIdx.x, blksz = blockDim.x,
        idx = blockIdx.x * (blksz << 1) + tid;
    __shared__ uint32_t cache[1024];
    cache[tid] = idx < N ? src[idx] : 0;
    cache[tid + blksz] = idx + blksz < N ? src[idx + blksz] : 0;
    for (int i = BLKSZ; i && tid < i; i >>= 1) {
        __syncthreads();
        cache[tid] += cache[tid + i];
    }
    if (!tid)
        src[blockIdx.x] = cache[0];
}

static __global__ void print_result(uint32_t *C)
{
    printf("%u\n", *C);
}

static inline void fast_add(int N, uint32_t *src)
{
#define divceil(a, b) (a + b - 1) / b
    const int blksz = 512;
    for (int rem = N; rem > 1; rem = divceil(rem, (blksz << 1)))
        _add_internal<<<dim3(divceil(rem, (blksz << 1))), dim3(blksz)>>>(src,
                                                                         rem);
#undef divceil
}

int main(int argc, char *argv[])
{
    uint32_t *cuC, key1, key2, N;
    cudaMalloc(&cuC, MAXN * sizeof(uint32_t));
    while (scanf("%d %u %u", &N, &key1, &key2) == 3) {
        dot_product<<<dim3(GRIDSZ), dim3(BLKSZ)>>>(cuC, key1, key2, N);
        fast_add(GRIDSZ * BLKSZ, cuC);
        print_result<<<dim3(1), dim3(1)>>>(cuC);
    }
    cudaFree(cuC);
    return 0;
}
