#include "kernels.h"

#include <cmath>
#include <algorithm>
#include <omp.h>

void rmsnorm(float* out, const float* x, const float* weight, int size, float eps) {
    float sum_squares = 0.0f;

    #pragma omp simd reduction(+:sum_squares)
    for (int i = 0; i < size; ++i) {
        sum_squares += x[i] * x[i];
    }

    float rms = std::sqrt(sum_squares / size + eps);
    float scale = 1.0f / rms;

    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        out[i] = (x[i] * scale) * weight[i];
    }
}




