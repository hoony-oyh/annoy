//
// Created by Young Hun Oh on 12/4/21.
//

#ifndef ANNOY_KERNEL_H
#define ANNOY_KERNEL_H

/* ==================================================================================================================
 * Kernel for vector substraction
================================================================================================================== */
template<>
inline void substract<float>(const float* x, const float* y, float* z, int f) {
    const int batch_size = 32;
    if (f > batch_size) {
        for (; f > batch_size-1; f-=batch_size) {
            __m256 t1 = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            __m256 t2 = _mm256_sub_ps(_mm256_loadu_ps(x+8), _mm256_loadu_ps(y+8));
            __m256 t3 = _mm256_sub_ps(_mm256_loadu_ps(x+16), _mm256_loadu_ps(y+16));
            __m256 t4 = _mm256_sub_ps(_mm256_loadu_ps(x+24), _mm256_loadu_ps(y+24));

            _mm256_storeu_ps(z, t1);
            _mm256_storeu_ps(z+8, t2);
            _mm256_storeu_ps(z+16, t3);
            _mm256_storeu_ps(z+24, t4);

            x += batch_size;
            y += batch_size;
            z += batch_size;
        }
    }
    if (f > 7) {
        for (; f > 7; f -= 8) {
            __m256 t = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            _mm256_storeu_ps(z, t);
            x += 8;
            y += 8;
            z += 8;
        }
    }
    // Don't forget the remaining values.
    for (; f > 0; f--) {
        (*z) = (*x) - (*y);
        x++;
        y++;
        z++;
    }
}

/* ==================================================================================================================
 * Kernel for vector scaling
================================================================================================================== */
template<>
inline void scale<float>(const float* x, float *z, float scale, int f) {
    const int batch_size = 64;
    __m256 s = _mm256_set1_ps(scale);
    if (f > batch_size) {
        for (; f > batch_size-1; f-=batch_size) {
            __m256 t1 = _mm256_mul_ps(_mm256_loadu_ps(x), s);
            __m256 t2 = _mm256_mul_ps(_mm256_loadu_ps(x+8), s);
            __m256 t3 = _mm256_mul_ps(_mm256_loadu_ps(x+16), s);
            __m256 t4 = _mm256_mul_ps(_mm256_loadu_ps(x+24), s);
            __m256 t5 = _mm256_mul_ps(_mm256_loadu_ps(x+32), s);
            __m256 t6 = _mm256_mul_ps(_mm256_loadu_ps(x+40), s);
            __m256 t7 = _mm256_mul_ps(_mm256_loadu_ps(x+48), s);
            __m256 t8 = _mm256_mul_ps(_mm256_loadu_ps(x+56), s);

            _mm256_storeu_ps(z, t1);
            _mm256_storeu_ps(z+8, t2);
            _mm256_storeu_ps(z+16, t3);
            _mm256_storeu_ps(z+24, t4);
            _mm256_storeu_ps(z+32, t5);
            _mm256_storeu_ps(z+40, t6);
            _mm256_storeu_ps(z+48, t7);
            _mm256_storeu_ps(z+56, t8);

            x += batch_size;
            z += batch_size;
        }
    }
    // Don't forget the remaining values.
    if (f > 7) {
        for (; f > 7; f -= 8) {
            __m256 t = _mm256_mul_ps(_mm256_loadu_ps(x), s);
            _mm256_storeu_ps(z, t);
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

/* ==================================================================================================================
 * Kernel for vector scaling
================================================================================================================== */
template<>
inline void aXplusbY<float>(const float* x, const float *y, float *z, float a, float b, int f) {
    const int batch_size = 64;
    __m256 _a = _mm256_set1_ps(a);
    __m256 _b = _mm256_set1_ps(b);

    if (f > batch_size) {
        for (; f > batch_size-1; f-=batch_size) {
            __m256 t1 = _mm256_fmadd_ps(_mm256_loadu_ps(y), _b, _mm256_mul_ps(_mm256_loadu_ps(x), _a));
            __m256 t2 = _mm256_fmadd_ps(_mm256_loadu_ps(y+8), _b, _mm256_mul_ps(_mm256_loadu_ps(x+8), _a));
            __m256 t3 = _mm256_fmadd_ps(_mm256_loadu_ps(y+16), _b, _mm256_mul_ps(_mm256_loadu_ps(x+16), _a));
            __m256 t4 = _mm256_fmadd_ps(_mm256_loadu_ps(y+24), _b, _mm256_mul_ps(_mm256_loadu_ps(x+24), _a));
            __m256 t5 = _mm256_fmadd_ps(_mm256_loadu_ps(y+32), _b, _mm256_mul_ps(_mm256_loadu_ps(x+32), _a));
            __m256 t6 = _mm256_fmadd_ps(_mm256_loadu_ps(y+40), _b, _mm256_mul_ps(_mm256_loadu_ps(x+40), _a));
            __m256 t7 = _mm256_fmadd_ps(_mm256_loadu_ps(y+48), _b, _mm256_mul_ps(_mm256_loadu_ps(x+48), _a));
            __m256 t8 = _mm256_fmadd_ps(_mm256_loadu_ps(y+56), _b, _mm256_mul_ps(_mm256_loadu_ps(x+56), _a));

            _mm256_storeu_ps(z, t1);
            _mm256_storeu_ps(z+8, t2);
            _mm256_storeu_ps(z+16, t3);
            _mm256_storeu_ps(z+24, t4);
            _mm256_storeu_ps(z+32, t5);
            _mm256_storeu_ps(z+40, t6);
            _mm256_storeu_ps(z+48, t7);
            _mm256_storeu_ps(z+56, t8);

            x += batch_size;
            y += batch_size;
            z += batch_size;
        }
    }
    // Don't forget the remaining values.
    if (f > 7) {
        for (; f > 7; f -= 8) {
            __m256 t = _mm256_fmadd_ps(_mm256_loadu_ps(y), _b, _mm256_mul_ps(_mm256_loadu_ps(x), _a));
            _mm256_storeu_ps(z, t);
            x += 8;
            y += 8;
            z += 8;
        }
    }
    // Don't forget the remaining values.
    for (; f > 0; f--) {
        (*z) = (*x) * a + (*y) * b;
        x++;
        y++;
        z++;
    }
}

/* ==================================================================================================================
 * Kernel for vector-vector product
================================================================================================================== */
// Horizontal single sum of 256bit vector.
inline float hsum256_ps_avx(__m256 v) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float short_dot_kernel(const float* x, const float *y, int f) {
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

inline float binary_tree_dot_kernel(const float* x, const float* y) {
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

template<>
inline float dot<float>(const float* x, const float *y, int f) {
    float result = 0;
    if (f > 63) {
        for (; f > 63; f-=64) {
            result += binary_tree_dot_kernel(x, y);
            x += 64;
            y += 64;
        }
    }
    result += short_dot_kernel(x, y, f);
    return result;
}

#endif //ANNOY_KERNEL_H
