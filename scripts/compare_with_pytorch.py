#!/usr/bin/env python3
"""
Compares C++ kernel outputs with PyTorch ground truth.
"""

import os
import subprocess
import json
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
RUNNER_SRC = os.path.join(ROOT_DIR, "test", "parity_runner.cpp")
RUNNER_BIN = os.path.join(BUILD_DIR, "parity_runner")

def compile_runner():
    os.makedirs(BUILD_DIR, exist_ok=True)
    # Ensure kernels.o is up to date
    subprocess.check_call(["make", "build/obj/kernels.o"], cwd=ROOT_DIR)
    
    # Get libomp prefix
    omp_prefix = subprocess.check_output(["brew", "--prefix", "libomp"]).decode().strip()

    cmd = [
        "clang++",
        "-std=c++17",
        "-O3",
        f"-I{os.path.join(ROOT_DIR, 'include')}",
        "-Xpreprocessor",
        "-fopenmp",
        f"-I{omp_prefix}/include",
        RUNNER_SRC,
        os.path.join(BUILD_DIR, "obj", "kernels.o"),
        "-o",
        RUNNER_BIN,
        f"-L{omp_prefix}/lib",
        "-lomp",
    ]
    subprocess.check_call(cmd)

def run_cpp():
    output = subprocess.check_output([RUNNER_BIN]).decode().strip()
    data = {}
    for line in output.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k] = json.loads(v)
    return data

# ==============================================================================
# PyTorch Ground Truth Implementations
# ==============================================================================

def pytorch_rmsnorm(x, weight, eps=1e-5):
    variance = torch.mean(x ** 2)
    return (x * torch.rsqrt(variance + eps)) * weight

def pytorch_matmul(x, w):
    return torch.matmul(w, x)

def pytorch_softmax(x):
    return F.softmax(x, dim=-1)

def pytorch_silu(x):
    return F.silu(x)

def pytorch_swiglu(hb1, hb2):
    return F.silu(hb1) * hb2

def pytorch_accum(a, b):
    return a + b

def pytorch_rope(vec, pos, head_size, n_heads):
    # vec shape: (n_heads, head_size)
    vec = vec.view(n_heads, head_size).clone()
    for h in range(n_heads):
        for i in range(0, head_size, 2):
            freq = 1.0 / (10000.0 ** (i / head_size))
            val = pos * freq
            cos_val = torch.cos(torch.tensor(val))
            sin_val = torch.sin(torch.tensor(val))
            v0 = vec[h, i].clone()
            v1 = vec[h, i + 1].clone()
            vec[h, i] = v0 * cos_val - v1 * sin_val
            vec[h, i + 1] = v0 * sin_val + v1 * cos_val
    return vec.view(-1)

def main():
    print("=" * 70)
    print(" Compiling and Running C++ Kernels vs. PyTorch Ground Truth")
    print("=" * 70)

    compile_runner()
    cpp_results = run_cpp()

    test_cases = []

    # 1. RMSNorm
    x_rms = torch.tensor([-1.5, 2.0, 0.5, -0.2, 3.1, -2.4, 1.1, 0.0], dtype=torch.float32)
    w_rms = torch.tensor([0.8, 1.2, 1.0, 0.5, 1.5, 0.9, 1.1, 1.0], dtype=torch.float32)
    torch_rms = pytorch_rmsnorm(x_rms, w_rms)
    test_cases.append(("RMSNorm", torch_rms, torch.tensor(cpp_results["rmsnorm_out"])))

    # 2. Matmul
    x_mm = torch.tensor([1.2, -0.8, 2.5, 0.4], dtype=torch.float32)
    w_mm = torch.tensor([
        [ 0.5, -0.2,  1.1,  0.9],
        [-1.0,  0.3,  0.7, -0.4],
        [ 0.2,  0.8, -0.5,  1.3]
    ], dtype=torch.float32)
    torch_mm = pytorch_matmul(x_mm, w_mm)
    test_cases.append(("Matmul (GEMV)", torch_mm, torch.tensor(cpp_results["matmul_out"])))

    # 3. Softmax
    x_sm = torch.tensor([2.1, -0.5, 3.4, 0.2, -1.8, 4.0], dtype=torch.float32)
    torch_sm = pytorch_softmax(x_sm)
    test_cases.append(("Softmax", torch_sm, torch.tensor(cpp_results["softmax_out"])))

    # 4. SiLU
    x_silu = torch.tensor([-3.0, -1.0, 0.0, 1.0, 2.5, 5.0], dtype=torch.float32)
    torch_silu = pytorch_silu(x_silu)
    test_cases.append(("SiLU", torch_silu, torch.tensor(cpp_results["silu_out"])))

    # 5. SwiGLU
    hb1 = torch.tensor([-2.0,  0.5, 1.2, -0.8,  3.0, 0.0], dtype=torch.float32)
    hb2 = torch.tensor([ 1.5, -1.0, 2.0,  0.4, -0.5, 4.2], dtype=torch.float32)
    torch_swiglu = pytorch_swiglu(hb1, hb2)
    test_cases.append(("SwiGLU", torch_swiglu, torch.tensor(cpp_results["swiglu_out"])))

    # 6. Accum
    a = torch.tensor([ 1.0, -2.5, 3.2, 0.0, -1.1], dtype=torch.float32)
    b = torch.tensor([-0.5,  2.5, 1.0, 4.3,  1.1], dtype=torch.float32)
    torch_accum = pytorch_accum(a, b)
    test_cases.append(("Accum (Residual)", torch_accum, torch.tensor(cpp_results["accum_out"])))

    # 7. RoPE (Q & K)
    q = torch.tensor([1.0, 2.0, 3.0, 4.0,  -1.0, 0.5, 2.5, -3.0], dtype=torch.float32)
    k = torch.tensor([0.8, -1.2, 2.0, 1.5], dtype=torch.float32)
    torch_rope_q = pytorch_rope(q, pos=7, head_size=4, n_heads=2)
    torch_rope_k = pytorch_rope(k, pos=7, head_size=4, n_heads=1)
    test_cases.append(("RoPE (Query)", torch_rope_q, torch.tensor(cpp_results["rope_q_out"])))
    test_cases.append(("RoPE (Key)", torch_rope_k, torch.tensor(cpp_results["rope_k_out"])))

    # ==============================================================================
    # Evaluation Table
    # ==============================================================================
    all_passed = True
    print(f"{'Kernel Name':<20} | {'Max Abs Diff':<15} | {'Tolerance':<10} | {'Status':<8}")
    print("-" * 70)

    for name, expected, actual in test_cases:
        max_diff = torch.max(torch.abs(expected - actual)).item()
        tol = 1e-5
        passed = max_diff <= tol
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"{name:<20} | {max_diff:<15.2e} | {tol:<10.1e} | {status:<8}")

    print("=" * 70)
    if all_passed:
        print("ALL KERNELS MATCH PYTORCH GROUND TRUTH WITH NUMERICAL PARITY!")
    else:
        print("SOME KERNELS DIVERGED FROM PYTORCH GROUND TRUTH!")
    print("=" * 70)

if __name__ == "__main__":
    main()
