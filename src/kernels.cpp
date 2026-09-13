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


void matmul(float* out, const float* x, const float* w, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;

        const float* w_row = w + (i * n);

        #pragma omp simd reduction(+:val)
        for (int j = 0; j < n; ++j) {
            val += w_row[j] * x[j];
        }

        out[i] = val;
    }
}

void softmax(float* x, int size) {
    // Pass 1: Find the max value to prevent float overflow
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // Pass 2: Exponentiate shifted values and sum them
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Pass 3: Normalize so all elements sum to 1.0
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

void silu(float* x, int size) {
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        x[i] = silu_scalar(x[i]);
    }
}

void swiglu(float* out, const float* hb1, const float* hb2, int hidden_dim) {
    #pragma omp simd
    for (int i = 0; i < hidden_dim; ++i) {
        out[i] = silu_scalar(hb1[i]) * hb2[i];
    }
}

void accum(float* a, const float* b, int size) {
    #pragma omp simd
    for (int i = 0; i < size; ++i) {
        a[i] += b[i];
    }
}

void apply_rope(float* q, float* k, int pos, int head_size, int n_heads, int n_kv_heads) {
    // 1. Rotate the Query heads
    for (int h = 0; h < n_heads; ++h) {
        float* q_head = q + (h * head_size);

        for (int i = 0; i < head_size; i += 2) {
            // Frequency for this specific pair of dimensions
            float freq = 1.0f / std::pow(10000.0f, (float)i / (float)head_size);
            float val = pos * freq;
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            // Read the pair (x, y)
            float q0 = q_head[i];
            float q1 = q_head[i + 1];

            // 2D rotation
            q_head[i]     = q0 * cos_val - q1 * sin_val;
            q_head[i + 1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // 2. Rotate the Key heads
    for (int h = 0; h < n_kv_heads; ++h) {
        float* k_head = k + (h * head_size);

        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / std::pow(10000.0f, (float)i / (float)head_size);
            float val = pos * freq;
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            float k0 = k_head[i];
            float k1 = k_head[i + 1];

            k_head[i]     = k0 * cos_val - k1 * sin_val;
            k_head[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}






