//
// Created by Young Hun Oh on 11/26/21.
//
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>    // std::sort
#include "rdtsc.h"

#include "dot.h"

using namespace std;

tsc_counter c0, c1;
long long cycle_sum;

void print_vectors(float **v, vector<int> indices, int f){
    for (int i=0; i < indices.size(); i++){
        int idx = indices[i];
        printf("%d: ", idx);
        for (int j = 0; j != f; ++j){
            printf("%f ", v[idx][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void correction_check(float* ref, float* your, int n) {
    int correctFlag = 1;
    for(int i = 0; i != n; ++i){
        if (your[i] - ref[i] > 0.005 || your[i] - ref[i] < -0.005 ){
            correctFlag = 0;
            break;
        }
    }
    if (correctFlag)
        printf("Correction Check: Correct!\n");
    else {
        printf("Correction Check: Incorrect!\n");
        for (int i = 0; i != n; ++i){
            printf("%f ", ref[i]);
        }
        printf("\n");
        for (int i = 0; i != n; ++i){
            printf("%f ", your[i]);
        }
        printf("\n");
    }
}

inline float vanilla_dot(const float* x, const float* y, int f) {
    float s = 0;
    for (int z = 0; z < f; z++) {
        s += (*x) * (*y);
        x++;
        y++;
    }
    return s;
}

inline void vanilla_matrix_dot(float **v, float *n, float *o, vector<int> indices, int f) {
    for (int i = 0; i != indices.size(); ++i) {
        int idx = indices[i];
        o[i] = vanilla_dot(v[idx], n, f);
    }
}

inline void simd_matrix_dot_kernel(float **v, float *n, float *o, vector<int> indices, int f, int start_idx) {
    float *x1 = v[indices[start_idx]];
    float *x2 = v[indices[start_idx+1]];
    float *x3 = v[indices[start_idx+2]];
    float *x4 = v[indices[start_idx+3]];
    float *x5 = v[indices[start_idx+4]];
    float *x6 = v[indices[start_idx+5]];
    float *x7 = v[indices[start_idx+6]];
    float *x8 = v[indices[start_idx+7]];
    float *x9 = v[indices[start_idx+8]];
    float *x10 = v[indices[start_idx+9]];

    __m256 t1 = _mm256_setzero_ps();
    __m256 t2 = _mm256_setzero_ps();
    __m256 t3 = _mm256_setzero_ps();
    __m256 t4 = _mm256_setzero_ps();
    __m256 t5 = _mm256_setzero_ps();
    __m256 t6 = _mm256_setzero_ps();
    __m256 t7 = _mm256_setzero_ps();
    __m256 t8 = _mm256_setzero_ps();
    __m256 t9 = _mm256_setzero_ps();
    __m256 t10 = _mm256_setzero_ps();

    float o1, o2, o3, o4, o5, o6, o7, o8, o9, o10;

    if (f>7) {
        RDTSC(c0);
        for (; f > 7; f -= 8) {
            __m256 n_chunk = _mm256_loadu_ps(n);
            __m256 v1 = _mm256_loadu_ps(x1);
            __m256 v2 = _mm256_loadu_ps(x2);
            __m256 v3 = _mm256_loadu_ps(x3);
            __m256 v4 = _mm256_loadu_ps(x4);
            __m256 v5 = _mm256_loadu_ps(x5);

            t1 = _mm256_fmadd_ps(v1, n_chunk, t1);
            t2 = _mm256_fmadd_ps(v2, n_chunk, t2);
            t3 = _mm256_fmadd_ps(v3, n_chunk, t3);
            t4 = _mm256_fmadd_ps(v4, n_chunk, t4);
            t5 = _mm256_fmadd_ps(v5, n_chunk, t5);
            t6 = _mm256_fmadd_ps(_mm256_loadu_ps(x6), n_chunk, t6);
            t7 = _mm256_fmadd_ps(_mm256_loadu_ps(x7), n_chunk, t7);
            t8 = _mm256_fmadd_ps(_mm256_loadu_ps(x8), n_chunk, t8);
            t9 = _mm256_fmadd_ps(_mm256_loadu_ps(x9), n_chunk, t9);
            t10 = _mm256_fmadd_ps(_mm256_loadu_ps(x10), n_chunk, t10);

            x1 += 8;
            x2 += 8;
            x3 += 8;
            x4 += 8;
            x5 += 8;
            x6 += 8;
            x7 += 8;
            x8 += 8;
            x9 += 8;
            x10 += 8;
            n += 8;
        }
        RDTSC(c1);
        cycle_sum += (COUNTER_DIFF(c1, c0, CYCLES));

        o1 = hsum256_ps_avx(t1);
        o2 = hsum256_ps_avx(t2);
        o3 = hsum256_ps_avx(t3);
        o4 = hsum256_ps_avx(t4);
        o5 = hsum256_ps_avx(t5);
        o6 = hsum256_ps_avx(t6);
        o7 = hsum256_ps_avx(t7);
        o8 = hsum256_ps_avx(t8);
        o9 = hsum256_ps_avx(t9);
        o10 = hsum256_ps_avx(t10);
    }
    for (; f > 0; f--) {
        o1 += *x1 * *n;
        o2 += *x2 * *n;
        o3 += *x3 * *n;
        o4 += *x4 * *n;
        o5 += *x5 * *n;
        o6 += *x6 * *n;
        o7 += *x7 * *n;
        o8 += *x8 * *n;
        o9 += *x9 * *n;
        o10 += *x10 * *n;

        x1 ++;
        x2 ++;
        x3 ++;
        x4 ++;
        x5 ++;
        x6 ++;
        x7 ++;
        x8 ++;
        x9 ++;
        x10 ++;
        n++;
    }
    o[start_idx] = o1;
    o[start_idx+1] = o2;
    o[start_idx+2] = o3;
    o[start_idx+3] = o4;
    o[start_idx+4] = o5;
    o[start_idx+5] = o6;
    o[start_idx+6] = o7;
    o[start_idx+7] = o8;
    o[start_idx+8] = o9;
    o[start_idx+9] = o10;
}

inline void simd_sequential_dot(float **v, float *n, float *o, vector<int> indices, int f) {
    for (int i=0; i<indices.size(); i++) {
        o[i] = short_simd_dot(v[indices[i]], n, f);
    }
}

inline void simd_matrix_dot(float **v, float *n, float *o, vector<int> indices, int f) {
    int nv = indices.size();
    int kernel_size = 10;
    int i = 0;
    for (; nv > kernel_size-1; nv -= kernel_size) {
        simd_matrix_dot_kernel(v, n, o, indices, f, i);
        i += kernel_size;
    }
    for (; nv > 0; nv--) {
        o[i] = long_simd_dot(v[indices[i]], n, f);
        i++;
    }
}


int main(int argc, char **argv) {
    int runs = atoi(argv[1]);
    int f = atoi(argv[2]);


    // allocate and initialize the vectors randomly
    float **v;
    float *n;
    float *o, *o_ref;
    int vec_num = 5000;
    int n_items = 10;
    posix_memalign((void**) &v, 64, vec_num * sizeof(float*));
    posix_memalign((void**) &n, 64, f * sizeof(float));
    posix_memalign((void**) &o, 64, n_items * sizeof(float));
    posix_memalign((void**) &o_ref, 64, n_items * sizeof(float));

    for (int i = 0; i < vec_num; i++) {
        posix_memalign((void**) &v[i], 64, f * sizeof(float));
        for (int j = 0; j != f; ++j){
            v[i][j] = ((float) rand())/ ((float) RAND_MAX);
        }
    }
    for (int i = 0; i != f; ++i){
        n[i] = 0.5 - ((float) rand())/ ((float) RAND_MAX);
    }

    // pick random vectors
    vector<int> indices;
    while(true) {
        int random_idx = (int)rand() % vec_num;
        if (find(indices.begin(), indices.end(), random_idx) == indices.end()){
            indices.push_back(random_idx);
        }
        if (indices.size()==n_items)
            break;
    }
    std::sort(indices.begin(), indices.end());

    printf("======================================================\n");
    printf("Matrix-Vector Dot Product Kernel Profiling\n");
    printf("======================================================\n");
    printf("Testing dot product with feature dimension %d\n", f);
    printf("Measuring the dot product between %d vectors and one vector\n\n", n_items);

    printf("Vanilla matrix-vector dot product =====================================================================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        for (int j = 0; j != indices.size(); ++j){
            o_ref[j] = 0.0;
        }
        RDTSC(c0);
        vanilla_matrix_dot(v, n, o_ref, indices, f);
        RDTSC(c1);
        cycle_sum += (COUNTER_DIFF(c1, c0, CYCLES));
    }
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (2 * f * n_items * runs / (double) cycle_sum)));
    printf("\n");

    printf("SIMD sequential dot product ===========================================================================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        for (int j = 0; j != indices.size(); ++j){
            o[j] = 0.0;
        }
        RDTSC(c0);
        simd_sequential_dot(v, n, o, indices, f);
        RDTSC(c1);
        cycle_sum += (COUNTER_DIFF(c1, c0, CYCLES));
    }
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) (2 * f * n_items * runs / (double) cycle_sum)));
    correction_check(o_ref, o, n_items);
    printf("\n");

    printf("SIMD matrix-vector dot product ========================================================================\n");
    cycle_sum = 0;
    for(unsigned int i = 0; i != runs; ++i) {
        for (int j = 0; j != indices.size(); ++j){
            o[j] = 0.0;
        }
//        RDTSC(c0);
        simd_matrix_dot(v, n, o, indices, f);
//        RDTSC(c1);
//        cycle_sum += (COUNTER_DIFF(c1, c0, CYCLES));
    }
    printf("Average Dot Product time: %lf cycles\n", ((double) (cycle_sum / ((double) runs))));
    printf("Op/Cycle = %f\n", ((double) ((2*f-8)* n_items * runs / (double) cycle_sum)));
    correction_check(o_ref, o, n_items);
    printf("\n");



    // free the resource
    for (int i = 0; i < vec_num; i++) {
        free(v[i]);
    }
    free(v);

    return 0;
}