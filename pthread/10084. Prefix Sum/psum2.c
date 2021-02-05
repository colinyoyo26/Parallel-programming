#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

#define MAXN 10000005
#define MAXTHREADS 11

#define divceil(a, b) (((a) + (b) -1) / (b))

static uint32_t n, key, psum[MAXN], tcnt;
static pthread_barrier_t barrier;
typedef struct {
    uint32_t begin, end, val;
} info_t;
info_t info[MAXTHREADS + 1];

static inline uint32_t rotate_left(uint32_t x, uint32_t n)
{
    return (x << n) | (x >> (32 - n));
}
static inline uint32_t encrypt(uint32_t m, uint32_t key)
{
    return (rotate_left(m, key & 31) + key) ^ key;
}
void output(uint32_t presum[], int n);

void *calpsum(void *arg)
{
    int tid = *(int *) arg;
    uint32_t sum = 0;
    for (int i = info[tid].begin; i < info[tid].end; i++) {
        sum += encrypt(i, key);
        psum[i] = sum;
    }
    pthread_barrier_wait(&barrier);
    if (info[tid].begin == 1) {
        for (int i = 1; i <= tcnt; i++)
            info[i].val = psum[info[i - 1].end - 1] + info[i - 1].val;
    }
    pthread_barrier_wait(&barrier);
    for (uint32_t i = info[tid].begin; i < info[tid].end; i++)
        psum[i] += info[tid].val;
    return NULL;
}

int main()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_t threads[MAXTHREADS];
    int tid[MAXTHREADS];

    for (int i = 0; i < MAXTHREADS + 1; i++)
        CPU_SET(i, &cpuset);

    while (scanf("%d %u", &n, &key) == 2) {
        int size = divceil(n, (MAXTHREADS + 1));
        tcnt = divceil(n, size) - 1;
        pthread_barrier_init(&barrier, NULL, tcnt + 1);
        for (int i = 0; i < tcnt; i++) {
            info[i].begin = i * size + 1;
            info[i].end = info[i].begin + size;
            tid[i] = i;
            pthread_create(&threads[i], NULL, calpsum, (void *) &tid[i]);
        }
        info[tcnt].begin = tcnt * size + 1;
        info[tcnt].end = n + 1;
        calpsum((void *) &tcnt);

        for (int i = 0; i < tcnt; i++)
            pthread_join(threads[i], NULL);
        output(psum, n);
    }
    return 0;
}
