//
// Created by Young Hun Oh on 12/3/21.
//
#include <stdio.h>
#include <stdlib.h>
#include "rdtsc.h"

#include <immintrin.h>

template<typename T>
inline void scale(const T* x, T* z, T scale, int f) {
    for (int i = 0; i < f; i++) {
        (*z) = (*x) * scale;
        x++;
        z++;
    }
}

inline void simd_scale(const float* x, float* z, float scale, int f) {
    __m256 s = _mm256_set1_ps(scale);
    if (f > 7) {
        for (; f > 7; f -= 8) {
            __m256 t = _mm256_mul_ps(_mm256_loadu_ps(x), s);
            _mm256_store_ps(z, t);
            x += 8;
            z += 8;
        }
    }
    // Don't forget the remaining values.
    for (; f > 0; f--) {
        (*z) = (*x) * scale;
        x++;
        z++;
    }
}

inline void long_simd_scale(const float* x, float* z, float scale, int f) {
    float result = 0;
    const int batch_size = 64;
    __m256 s = _mm256_set1_ps(scale);
    if (f > batch_size) {
        for (; f > batch_size-1; f-=batch_size) {
            __m256 t1 = _mm256_loadu_ps(x);
            __m256 t2 = _mm256_loadu_ps(x+8);
            __m256 t3 = _mm256_loadu_ps(x+16);
            __m256 t4 = _mm256_loadu_ps(x+24);
            __m256 t5 = _mm256_loadu_ps(x+32);
            __m256 t6 = _mm256_loadu_ps(x+40);
            __m256 t7 = _mm256_loadu_ps(x+48);
            __m256 t8 = _mm256_loadu_ps(x+56);

            t1 = _mm256_mul_ps(t1, s);
            t2 = _mm256_mul_ps(t2, s);
            t3 = _mm256_mul_ps(t3, s);
            t4 = _mm256_mul_ps(t4, s);
            t5 = _mm256_mul_ps(t5, s);
            t6 = _mm256_mul_ps(t6, s);
            t7 = _mm256_mul_ps(t7, s);
            t8 = _mm256_mul_ps(t8, s);

            _mm256_store_ps(z, t1);
            _mm256_store_ps(z+8, t2);
            _mm256_store_ps(z+16, t3);
            _mm256_store_ps(z+24, t4);
            _mm256_store_ps(z+32, t5);
            _mm256_store_ps(z+40, t6);
            _mm256_store_ps(z+48, t7);
            _mm256_store_ps(z+56, t8);

            x += batch_size;
            z += batch_size;
        }
    }
    simd_scale(x, z, scale, f);
}

void correction_check(float* ref, float* your, int f) {
    int correctFlag = 1;
    for(int i = 0; i != f; ++i){
        if (your[i] != ref[i]){
            printf("%f %f\n", your[i], ref[i]);
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
    float *a, *c_ref, *c;

    int runs = atoi(argv[1]);
    int f = atoi(argv[2]);

    tsc_counter t0, t1;
    long long cycle_sum;

    posix_memalign((void**) &a, 64, f * sizeof(float));
    posix_memalign((void**) &c, 64, f * sizeof(float));
    posix_memalign((void**) &c_ref, 64, f * sizeof(float));

    //initialize A
    for (int i = 0; i != f; ++i){
        a[i] = ((float) rand())/ ((float) RAND_MAX);
    }

    //initialize C
    for (int i = 0; i != f; ++i){
        c[i] = 0;
    }
    for (int i = 0; i != f; ++i){
        c_ref[i] = 0;
    }

    float s = ((float) rand())/ ((float) RAND_MAX);

    printf("======================================================\n");
    printf("Vector Scaling Kernel Profiling\n");
    printf("======================================================\n");
    printf("Testing vector scaling with feature dimension %d\n\n", f);

    printf("Vanilla vector scaling =========================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        scale(a, c_ref, s, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average Scaling time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * runs / (double) cycle_sum)));

    printf("\nSIMD vector scaling ============================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        simd_scale(a, c, s, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average Scaling time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * runs / (double) cycle_sum)));
    correction_check(c_ref, c, f);

    printf("\nSIMD vector scaling (long)======================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        long_simd_scale(a, c, s, f);
        RDTSC(t1);
        cycle_sum += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average Scaling time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (f * runs / (double) cycle_sum)));
    correction_check(c_ref, c, f);

    // clean-up
    free(a);
    free(c);
    printf("\n\n");
}