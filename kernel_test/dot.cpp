//
// Created by Young Hun Oh on 11/12/21.
//
#include <stdio.h>
#include <stdlib.h>
#include "rdtsc.h"

#include <immintrin.h>

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

inline float hsum256_ps_avx(__m256 v) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float short_simd_dot(const float* x, const float *y, int f) {
    float result = 0;
    if (f > 7) {
        __m256 d = _mm256_setzero_ps();
        for (; f > 7; f -= 8) {
            d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
            x += 8;
            y += 8;
        }
        // Sum all floats in dot register.
        result += hsum256_ps_avx(d);
    }
    // Don't forget the remaining values.
    for (; f > 0; f--) {
        result += *x * *y;
        x++;
        y++;
    }
    return result;
}

/*
 * Kernel for efficient dot product of high-dimensional vectors
 *
 */
inline float binary_tree_dot(const float* x, const float *y) {
    __m256 t1 = _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
    __m256 t2 = _mm256_mul_ps(_mm256_loadu_ps(x+8), _mm256_loadu_ps(y+8));
    __m256 t3 = _mm256_mul_ps(_mm256_loadu_ps(x+16), _mm256_loadu_ps(y+16));
    __m256 t4 = _mm256_mul_ps(_mm256_loadu_ps(x+24), _mm256_loadu_ps(y+24));
    __m256 t5 = _mm256_mul_ps(_mm256_loadu_ps(x+32), _mm256_loadu_ps(y+32));
    __m256 t6 = _mm256_mul_ps(_mm256_loadu_ps(x+40), _mm256_loadu_ps(y+40));
    __m256 t7 = _mm256_mul_ps(_mm256_loadu_ps(x+48), _mm256_loadu_ps(y+48));
    __m256 t8 = _mm256_mul_ps(_mm256_loadu_ps(x+56), _mm256_loadu_ps(y+56));

    t1 = _mm256_add_ps(t1, t2);
    t3 = _mm256_add_ps(t3, t4);
    t5 = _mm256_add_ps(t5, t6);
    t7 = _mm256_add_ps(t7, t8);

    t1 = _mm256_add_ps(t1, t3);
    t5 = _mm256_add_ps(t5, t7);

    t1 = _mm256_add_ps(t1, t5);

    float result = hsum256_ps_avx(t1);
    return result;
}

inline float long_simd_dot(const float* x, const float *y, int f) {
    float result = 0;
    if (f > 63) {
        for (; f > 63; f-=64) {
            result += binary_tree_dot(x, y);
            x += 64;
            y += 64;
        }
    }
    result += short_simd_dot(x, y, f);
    return result;
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