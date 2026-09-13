#include "kernels.h"
#include <iostream>
#include <vector>
#include <iomanip>

void print_array(const std::string& name, const float* arr, int size) {
    std::cout << name << ":[";
    for (int i = 0; i < size; ++i) {
        std::cout << std::setprecision(8) << arr[i];
        if (i + 1 < size) std::cout << ",";
    }
    std::cout << "]\n";
}

int main() {
    // 1. RMSNorm
    {
        std::vector<float> x = {-1.5f, 2.0f, 0.5f, -0.2f, 3.1f, -2.4f, 1.1f, 0.0f};
        std::vector<float> weight = {0.8f, 1.2f, 1.0f, 0.5f, 1.5f, 0.9f, 1.1f, 1.0f};
        std::vector<float> out(8, 0.0f);
        rmsnorm(out.data(), x.data(), weight.data(), 8, 1e-5f);
        print_array("rmsnorm_out", out.data(), 8);
    }

    // 2. MatMul
    {
        int n = 4, d = 3;
        std::vector<float> x = {1.2f, -0.8f, 2.5f, 0.4f};
        std::vector<float> w = {
            0.5f, -0.2f,  1.1f,  0.9f,
           -1.0f,  0.3f,  0.7f, -0.4f,
            0.2f,  0.8f, -0.5f,  1.3f
        };
        std::vector<float> out(d, 0.0f);
        matmul(out.data(), x.data(), w.data(), n, d);
        print_array("matmul_out", out.data(), d);
    }

    // 3. Softmax
    {
        std::vector<float> x = {2.1f, -0.5f, 3.4f, 0.2f, -1.8f, 4.0f};
        softmax(x.data(), x.size());
        print_array("softmax_out", x.data(), x.size());
    }

    // 4. SiLU
    {
        std::vector<float> x = {-3.0f, -1.0f, 0.0f, 1.0f, 2.5f, 5.0f};
        silu(x.data(), x.size());
        print_array("silu_out", x.data(), x.size());
    }

    // 5. SwiGLU
    {
        std::vector<float> hb1 = {-2.0f,  0.5f, 1.2f, -0.8f,  3.0f, 0.0f};
        std::vector<float> hb2 = { 1.5f, -1.0f, 2.0f,  0.4f, -0.5f, 4.2f};
        std::vector<float> out(6, 0.0f);
        swiglu(out.data(), hb1.data(), hb2.data(), 6);
        print_array("swiglu_out", out.data(), 6);
    }

    // 6. Accum
    {
        std::vector<float> a = { 1.0f, -2.5f, 3.2f, 0.0f, -1.1f};
        std::vector<float> b = {-0.5f,  2.5f, 1.0f, 4.3f,  1.1f};
        accum(a.data(), b.data(), a.size());
        print_array("accum_out", a.data(), a.size());
    }

    // 7. RoPE
    {
        int head_size = 4, n_heads = 2, n_kv_heads = 1, pos = 7;
        std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f,  -1.0f, 0.5f, 2.5f, -3.0f};
        std::vector<float> k = {0.8f, -1.2f, 2.0f, 1.5f};
        apply_rope(q.data(), k.data(), pos, head_size, n_heads, n_kv_heads);
        print_array("rope_q_out", q.data(), q.size());
        print_array("rope_k_out", k.data(), k.size());
    }

    return 0;
}
