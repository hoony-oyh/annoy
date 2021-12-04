//
// Created by Young Hun Oh on 11/23/21.
//
#include <stdio.h>
#include <stdlib.h>
#include "rdtsc.h"

#include <immintrin.h>

// _mm256_set1_ps(float a)
template<typename T>
inline void add(const T* x, const T* y, T* z, int f) {
    for (int i = 0; i < f; i++) {
        (*z) = (*x) + (*y);
        x++;
        y++;
        z++;
    }
}

inline void simd_add(const float* x, const float* y, float* z, int f) {
    if (f > 7) {
        for (; f > 7; f -= 8) {
            __m256 t = _mm256_add_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_store_ps(z, t);
            x += 8;
            y += 8;
            z += 8;
        }
    }
    // Don't forget the remaining values.
    for (; f > 0; f--) {
        (*z) = (*x) + (*y);
        x++;
        y++;
        z++;
    }
}

inline void long_simd_add(const float* x, const float* y, float* z, int f) {
    float result = 0;
    const int batch_size = 32;
    if (f > batch_size) {
        for (; f > batch_size-1; f-=batch_size) {
            __m256 t1 = _mm256_add_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            __m256 t2 = _mm256_add_ps(_mm256_loadu_ps(x+8), _mm256_loadu_ps(y+8));
            __m256 t3 = _mm256_add_ps(_mm256_loadu_ps(x+16), _mm256_loadu_ps(y+16));
            __m256 t4 = _mm256_add_ps(_mm256_loadu_ps(x+24), _mm256_loadu_ps(y+24));
//            __m256 t5 = _mm256_add_ps(_mm256_loadu_ps(x+32), _mm256_loadu_ps(y+32));
//            __m256 t6 = _mm256_add_ps(_mm256_loadu_ps(x+40), _mm256_loadu_ps(y+40));
//            __m256 t7 = _mm256_add_ps(_mm256_loadu_ps(x+48), _mm256_loadu_ps(y+48));
//            __m256 t8 = _mm256_add_ps(_mm256_loadu_ps(x+56), _mm256_loadu_ps(y+56));

            _mm256_store_ps(z, t1);
            _mm256_store_ps(z+8, t2);
            _mm256_store_ps(z+16, t3);
            _mm256_store_ps(z+24, t4);
//            _mm256_store_ps(z+32, t5);
//            _mm256_store_ps(z+40, t6);
//            _mm256_store_ps(z+48, t7);
//            _mm256_store_ps(z+56, t8);

            x += batch_size;
            y += batch_size;
            z += batch_size;
        }
    }
    simd_add(x, y, z, f);
}

void correction_check(float* ref, float* your, int f) {
    int correctFlag = 1;
    for(int i = 0; i != f; ++i){
        if (your[i] != ref[i]){
            correctFlag = 0;
            break;
        }
    }
    if (correctFlag)
        printf("Correction Check: Correct!\n");
    else
        printf("Correction Check: Incorrect!\n");
}

int main(int argc, char **argv) {
    float *a, *b, *c_ref, *c;

    int runs = atoi(argv[1]);
    int f = atoi(argv[2]);

    tsc_counter t0, t1;
    long long cycle_sum;

    posix_memalign((void**) &a, 64, f * sizeof(float));
    posix_memalign((void**) &b, 64, f * sizeof(float));
    posix_memalign((void**) &c, 64, f * sizeof(float));
    posix_memalign((void**) &c_ref, 64, f * sizeof(float));

    //initialize A
    for (int i = 0; i != f; ++i){
        a[i] = ((float) rand())/ ((float) RAND_MAX);
    }

    //initialize B
    for (int i = 0; i != f; ++i){
        b[i] = ((float) rand())/ ((float) RAND_MAX);
    }

    //initialize C
    for (int i = 0; i != f; ++i){
        c[i] = 0;
    }
    for (int i = 0; i != f; ++i){
        c_ref[i] = 0;
    }

    printf("======================================================\n");
    printf("Vector Addition Kernel Profiling\n");
    printf("======================================================\n");
    printf("Testing vector addition with feature dimension %d\n\n", f);

    printf("Vanilla vector addition =========================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        add(a, b, c_ref, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * runs / (double) cycle_sum)));

    printf("\nSIMD vector addition ============================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        simd_add(a, b, c, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * runs / (double) cycle_sum)));
    correction_check(c_ref, c, f);

    printf("\nSIMD vector addition (long)======================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        long_simd_add(a, b, c, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * runs / (double) cycle_sum)));
    correction_check(c_ref, c, f);

    // clean-up
    free(a);
    free(b);
    free(c);
    printf("\n\n");
}