#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

#define MAXN 10000005
#define MAXTHREADS 11

#define divceil(a, b) (((a) + (b) -1) / (b))

typedef struct {
    uint32_t begin, end, val;
    pthread_mutex_t lock;
} info_t;
static uint32_t n, key, psum[MAXN], tcnt;
static info_t info[MAXTHREADS + 1];
static pthread_barrier_t b1, b2, b3;

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
    do {
        if ((intptr_t) arg != (intptr_t) &tcnt)
            pthread_mutex_lock(&info[tid].lock);
        uint32_t sum = 0;
        for (int i = info[tid].begin; i < info[tid].end; i++) {
            sum += encrypt(i, key);
            psum[i] = sum;
        }
        pthread_barrier_wait(&b1);
        if (info[tid].begin == 1) {
            for (int i = 1; i <= tcnt; i++)
                info[i].val = psum[info[i - 1].end - 1] + info[i - 1].val;
        }
        pthread_barrier_wait(&b2);
        for (int i = info[tid].begin; i < info[tid].end; i++)
            psum[i] += info[tid].val;
        pthread_barrier_wait(&b3);
    } while ((intptr_t) arg != (intptr_t) &tcnt);
    return NULL;
}

int main()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_t threads[MAXTHREADS];
    int tid[MAXTHREADS];

    for (int i = 0; i < MAXTHREADS; i++) {
        CPU_SET(i, &cpuset);
        tid[i] = i;
        pthread_mutex_init(&info[i].lock, NULL);
        pthread_mutex_lock(&info[i].lock);
        pthread_create(&threads[i], NULL, calpsum, (void *) &tid[i]);
    }

    while (scanf("%d %u", &n, &key) == 2) {
        int size = divceil(n, (MAXTHREADS + 1));
        tcnt = divceil(n, size) - 1;
        pthread_barrier_init(&b1, NULL, tcnt + 1);
        pthread_barrier_init(&b2, NULL, tcnt + 1);
        pthread_barrier_init(&b3, NULL, tcnt + 1);
        for (int i = 0; i < tcnt; i++) {
            info[i].begin = i * size + 1;
            info[i].end = info[i].begin + size;
            pthread_mutex_unlock(&info[i].lock);
        }
        info[tcnt].begin = tcnt * size + 1;
        info[tcnt].end = n + 1;
        calpsum((void *) &tcnt);
        output(psum, n);
    }
    return 0;
}
