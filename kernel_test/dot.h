//
// Created by Young Hun Oh on 11/26/21.
//
#include <immintrin.h>

#ifndef ANNOY_DOT_H
#define ANNOY_DOT_H

inline float hsum256_ps_avx(__m256 v) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float mm256_cvtss_f32(__m256 v) {
    return _mm_cvtss_f32(_mm256_castps256_ps128(v));
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



#endif //ANNOY_DOT_H
