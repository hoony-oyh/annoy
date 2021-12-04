//
// Created by Young Hun Oh on 11/12/21.
//
#include <stdio.h>
#include <stdlib.h>
#include "rdtsc.h"

#include "dot.h"

template<typename T>
inline T vanilla_dot(const T* x, const T* y, int f) {
    T s = 0;
    for (int z = 0; z < f; z++) {
        s += (*x) * (*y);
        x++;
        y++;
    }
    return s;
}

int main(int argc, char **argv) {
    float *a, *b;

    int runs = atoi(argv[1]);
    int f = atoi(argv[2]);

    tsc_counter t0, t1;
    long long cycle_sum;

    printf("======================================================\n");
    printf("Dot Product Kernel Profiling\n");
    printf("======================================================\n");
    printf("Testing dot product with feature dimension %d\n\n", f);
    posix_memalign((void**) &a, 64, f * sizeof(float));
    posix_memalign((void**) &b, 64, f * sizeof(float));

    //initialize A
    for (int i = 0; i != f; ++i){
        a[i] = ((float) rand())/ ((float) RAND_MAX);
    }

    //initialize B
    for (int i = 0; i != f; ++i){
        b[i] = ((float) rand())/ ((float) RAND_MAX);
    }

    float x = 0;
    printf("Vanilla scalar dot product =========================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        x = vanilla_dot(a, b, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Dot product result: %f\n", x);
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * 2 * runs / (double) cycle_sum)));

    printf("\nSimple SIMD dot product ============================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        x = short_simd_dot(a, b, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Dot product result: %f\n", x);
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * 2 * runs / (double) cycle_sum)));

    printf("\nOur SIMD dot product ===============================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        x = long_simd_dot(a, b, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Dot product result: %f\n", x);
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * 2 * runs / (double) cycle_sum)));

    free(a);
    free(b);
    printf("\n\n");
}