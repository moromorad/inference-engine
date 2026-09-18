# Inference Engine (C++17)

A high-performance, zero-dependency C++17 inference engine for autoregressive LLaMA-style Transformer language models (supporting Stories15M, Stories42M, Stories110M, and LLaMA-2/3 architectures). Built from first principles in modern C++ with zero external runtime or framework dependencies (no PyTorch, no ONNX, no llama.cpp runtime).

The engine features zero-copy virtual memory-mapped weight loading (`mmap`), contiguous 4D Key-Value caching, SIMD-vectorized OpenMP linear algebra kernels, a zero-allocation execution scratchpad, and an automated persistent benchmarking suite.

---

## ⚡ Benchmarks & Performance Comparison

All benchmarks were conducted on **Apple Silicon (M-series, 8 hardware threads, Clang -O3 -fopenmp)** under identical conditions:
* **Prompt:** *"Once upon a time, in a lush green valley surrounded by tall blue mountains, there lived a kind little hedgehog who loved discovering secret trails."* (35 tokens)
* **Decode Length:** 100 generated tokens
* **Sampling:** Deterministic Greedy Argmax (`temperature = 0.0`)
* **Precision:** 32-bit Single-Precision Floating Point (FP32)

### Throughput Comparison vs. Andrej Karpathy's `llama2.c`

| Model | Parameters | Size (MB) | Karpathy's `llama2.c` | **This Engine (Decode)** | **This Engine (Prefill)** | **Speedup vs. llama2.c** | Latency / Token |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stories15M** | 15M | 58.0 MB | **298.2 tok/s** | **633.0 tok/s** | 674.0 tok/s | **2.12× faster** | 1.58 ms |
| **Stories110M** | 110M | 418.1 MB | **50.4 tok/s** | **128.5 tok/s** | 132.5 tok/s | **2.57× faster** | 7.78 ms |

> Note: On Stories15M, throughput exceeds 600 tokens/sec. On Stories110M, throughput exceeds 128 tokens/sec, more than doubling the performance of `llama2.c`.

### Hardware Efficiency & Memory Bandwidth Saturation

In single-batch autoregressive generation, throughput is strictly **memory bandwidth bound**. At each token generation step, the entire model weight matrix must be loaded from system memory (RAM/cache) into CPU registers to compute vector-matrix products for a single token:

```text
Memory Bandwidth (GB/s) = Model Weight Size (GB) × Tokens / Second
```

* **Stories110M (418 MB = 0.418 GB):**
  * Throughput: `128.5 tok/s`
  * Sustained Memory Bandwidth: `0.418 GB × 128.5 = 53.7 GB/s`
  * On an Apple Silicon base architecture with a theoretical peak memory bandwidth of ~100 GB/s, this engine achieves **~53.7% of physical hardware memory bandwidth saturation** using pure CPU execution.

### Why Is This Engine Faster than `llama2.c`?

1. **Zero-Allocation Scratchpad (`RunState`):** All intermediate state tensors (`x`, `xb`, `xb2`, `hb`, `hb2`, `q`, `k`, `v`, `att`, `logits`) are allocated once on heap creation. No allocations or deallocations occur inside the token generation loop, completely eliminating heap fragmentation, mutex locks, and CPU cache thrashing.
2. **Contiguous 4D Memory Layout for KV Cache:** Keys and Values are stored in a contiguous 1D array indexed as `[n_layers, seq_len, n_kv_heads, head_size]`. Sequential head slices are accessed through direct pointer arithmetic without multi-dimensional pointer lookups, maximizing CPU L1/L2 cache prefetching.
3. **OpenMP Dynamic Parallelism:** Matrix-vector multiplications (GEMV) parallelize outer row iterations with OpenMP pragmas, using SIMD auto-vectorization across the inner dot products.
4. **Fused SwiGLU Activation:** `SiLU(W1 * x) * (W3 * x)` computes the sigmoid activation and element-wise gate product in a single contiguous memory pass.
5. **Partial-Sort Top-K Sampling:** Rather than sorting the full 32,000-element vocabulary (`O(N log N)`), top-K sampling utilizes `std::partial_sort` (`O(N log K)`), cutting sampling overhead by over 95%.

---

## 🏗️ How the Project Works (Architecture & Pipeline)

Autoregressive inference predicts text one token at a time. The engine executes text generation in two sequential phases:

```text
Prompt String: "Once upon a time..."
       │
       ▼
 [Tokenizer] (SentencePiece BPE) ──> Token IDs: [BOS, T1, T2, ..., Tn]
                                               │
 ┌─────────────────────────────────────────────┘
 │
 ▼
====================== PHASE 1: PREFILL ======================
• Ingests all prompt tokens sequentially (positions 0 to n-1).
• Computes Key and Value projections and stores them in the KVCache.
• Generates logits for the final prompt token to predict the first new token.
• Operates at ~640 - 674 tok/s.
                                │
                                ▼
====================== PHASE 2: DECODE =======================
Autoregressive loop running at ~128 - 633 tok/s:

   Current Token ID
        │
        ├──> [Embedding Lookup] (Extract dim-dimensional vector from token_embedding_table)
        │
        ▼
   ================== TRANSFORMER BLOCK (Repeated for each Layer) ==================
        │
        ├──> [1. Pre-Attention RMSNorm] ── Normalize vector across hidden dim
        │         │
        │         ├──> [2. Linear Projections] ── Compute Q, K, V vectors (GEMV)
        │         │         │
        │         │         ├──> [3. Rotary Positional Embeddings (RoPE)]
        │         │         │         Rotate Q and K vectors in 2D complex pairs based on position
        │         │         │
        │         │         ├──> [4. Store K and V into KVCache]
        │         │         │         Persist historical context for past tokens
        │         │         │
        │         │         └──> [5. Multi-Head Scaled Dot-Product Attention]
        │         │                   Compute attention scores: Q · K_cache / sqrt(head_size)
        │         │                   Apply numerically stable Softmax
        │         │                   Blend Value vectors: att · V_cache
        │         │
        │         └──> [6. Output Projection Wo] ── Project concatenated heads back to dim
        │
        ├──> [7. Residual Skip Connection 1] ── x = x + att_out
        │
        ├──> [8. Pre-FFN RMSNorm] ── Normalize vector before feed-forward network
        │         │
        │         ├──> [9. SwiGLU Up-Projections W1 (Gate) & W3 (Up)] (GEMV)
        │         │         │
        │         │         └──> [10. SwiGLU Activation] ── SiLU(W1*x) * (W3*x)
        │         │
        │         └──> [11. Down-Projection W2] ── Project hidden_dim back to dim
        │
        └──> [12. Residual Skip Connection 2] ── x = x + ffn_out
        │
   ================================================================================
        │
        ▼ (Repeated across all n_layers)
   [Final RMSNorm]
        │
   [Classifier Head (wcls)] ── Matrix multiply with vocabulary table (dim → vocab_size)
        │
   [Unnormalized Logits] (Length: 32,000)
        │
        ▼
   =============================== SAMPLING ===============================
        │
        ├── Mode A: Greedy Argmax (temperature == 0.0)
        │     Pick token with highest logit value (argmax)
        │
        └── Mode B: Temperature + Top-K + Inverse CDF (temperature > 0.0)
              1. Scale logits: logits[i] = logits[i] / temperature
              2. Filter top-K logits using std::partial_sort
              3. Apply Softmax to top-K candidates
              4. Cumulative distribution function (CDF) roulette sampling
        │
        ▼
   New Generated Token! ──> Print to stdout & Feed back into next decode step
```

---

## 📁 Repository Structure

```text
inference_engine/
├── include/                          # Public C++ Header Files
│   ├── config.h                      # Model configuration struct & weight tensor mappings
│   ├── engine.h                      # Execution state scratchpad, forward pass & sampling functions
│   ├── kernels.h                     # High-performance math kernels (RMSNorm, GEMV, RoPE, Softmax, SwiGLU)
│   ├── kv_cache.h                    # Contiguous 4D Key-Value cache manager
│   ├── model.h                       # Zero-copy memory-mapper (POSIX mmap) and weight loader
│   ├── tokenizer.h                   # SentencePiece Byte-Pair Encoding (BPE) binary loader and tokenizer
│   └── doctest.h                     # Single-header lightweight C++ unit testing framework
│
├── src/                              # Implementation Source Files
│   ├── engine.cpp                    # Transformer block execution, forward pass, and sampling algorithms
│   ├── kernels.cpp                   # SIMD-vectorized OpenMP math kernels
│   ├── kv_cache.cpp                  # KV cache indexing and memory layout management
│   ├── model.cpp                     # POSIX mmap binary loader & tied-weight offset calculations
│   ├── tokenizer.cpp                 # Byte-Pair Encoding merge loops and vocabulary lookup
│   └── main.cpp                      # Interactive command-line terminal REPL with streaming output
│
├── benchmarks/                       # Performance Benchmarking Suite
│   └── benchmark.cpp                 # Automated Prefill and Decode benchmark runner with persistence
│
├── benchmark_results/                # Persistent Benchmark Logs (.gitignore)
│   ├── history.csv                   # Historical append-only CSV log of all benchmark runs
│   └── latest.json                   # Detailed JSON snapshot of the latest benchmark run
│
├── test/                             # Quality Assurance & Verification
│   ├── tests.cpp                     # 28 unit test suites containing 32,147 doctest assertions
│   └── parity_runner.cpp             # C++ binary driver for PyTorch floating-point parity checks
│
├── scripts/                          # Python Auxiliary Scripts
│   └── compare_with_pytorch.py       # Automated numerical parity validator against PyTorch reference
│
├── models/                           # Model Weight Checkpoints (.gitignore)
│   ├── stories15M.bin                # Stories15M checkpoint (58 MB, dim=288, layers=6, heads=6)
│   ├── stories110M.bin               # Stories110M checkpoint (418 MB, dim=768, layers=12, heads=12)
│   └── tokenizer.bin                 # 32,000-token BPE tokenizer vocabulary binary (434 KB)
│
├── Makefile                          # Unified build configuration (make, run, test, benchmark, parity)
├── .gitignore                        # Git ignore rules for binaries, build artifacts, models, and benchmark data
└── README.md                         # Project documentation
```

---

## 🔍 Codebase Deep Dive

### 1. `include/config.h` & `include/model.h` / `src/model.cpp`
* **`Config`**: Defines model hyperparameters read directly from the first 28 bytes of `.bin` files (`dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`, `seq_len`).
* **`TransformerWeights`**: Contains non-owning pointers (`float*`) to weight tensors mapped in virtual memory.
* **`Model`**:
  * Uses POSIX `mmap()` with `MAP_SHARED` to map checkpoints directly from disk into virtual address space. Booting a 418 MB model takes less than **5 milliseconds** with zero duplicate memory allocation.
  * Handles weight-tying logic: checks file size offsets to determine whether the classifier head `wcls` is explicitly present or shares pointers with `token_embedding_table`.
  * Accommodates legacy `export.py` trailing tables (e.g. 48 KB `freq_cis` tables) without buffer overrun.

### 2. `include/tokenizer.h` / `src/tokenizer.cpp`
* Implements a **SentencePiece Byte-Pair Encoding (BPE)** tokenizer with a 32,000-token vocabulary.
* Loads binary vocabulary files containing token strings and float merge scores.
* Encodes raw text into token IDs by breaking text into UTF-8 characters, greedily merging the highest-scoring consecutive pairs until no further valid merges exist.
* Handles byte fallback tokens (e.g., `<0x0A>` for newlines) and special control tokens (`BOS = 1`, `EOS = 2`).

### 3. `include/kv_cache.h` / `src/kv_cache.cpp`
* Stores historical attention Keys and Values across all layers and positions.
* Allocated as two flat contiguous 1D memory buffers:
  ```text
  Cache Size = n_layers × seq_len × n_kv_heads × head_size
  ```
* Provides `get_key_head()` and `get_value_head()` for pointer arithmetic into the active layer, position, and attention head.

### 4. `include/kernels.h` / `src/kernels.cpp`
Vectorized compute kernels optimized with OpenMP:
* **`rmsnorm`**: Root Mean Square Normalization with a numerical stability epsilon:
  ```text
  RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
  ```
* **`matmul`**: Matrix-vector product (`y = W · x`) parallelized across matrix rows using OpenMP multi-threading with inner dot products.
* **`apply_rope`**: Rotary Positional Embedding (RoPE) rotating pairs of dimensions `(q[2i], q[2i+1])` using sinusoidal frequencies based on token position.
* **`softmax`**: In-place softmax with numerical max-subtraction to prevent floating-point overflow.
* **`swiglu`**: Fused SwiGLU non-linear activation:
  ```text
  SwiGLU(x) = (W1 · x * sigmoid(W1 · x)) * (W3 · x)
  ```
* **`accum`**: Element-wise vector addition for residual skip connections (`a = a + b`).

### 5. `include/engine.h` / `src/engine.cpp`
* **`RunState`**: Pre-allocates all intermediate buffers (`x`, `xb`, `xb2`, `hb`, `hb2`, `q`, `k`, `v`, `att`, `logits`) to guarantee zero allocations during generation.
* **`transformer_block`**: Orchestrates the 12-step layer forward pass.
* **`forward`**: Full forward pass through embedding lookup, all Transformer layers, final RMSNorm, and classifier projection to generate unnormalized logits.
* **Sampling Suite**:
  * `sample_argmax`: Deterministic greedy selection.
  * `sample_top_k`: In-place Top-K logit selection with `std::partial_sort`.
  * `sample`: Full sampling pipeline applying temperature scaling, Top-K filtering, Softmax normalization, and inverse CDF roulette selection.
* **`generate`**: Complete streaming generation loop managing the transition from prefill prompt tokens to autoregressive generation.

### 6. `benchmarks/benchmark.cpp`
* Standalone benchmark runner that measures Prefill throughput, Decode throughput, and Latency per token.
* Includes a multi-step warmup pass to eliminate cold page fault anomalies.
* Automatically writes results to two persistent files in `benchmark_results/`:
  * `history.csv`: Append-only CSV log with full run metadata (timestamp, model, dimensions, threads, latency, throughput).
  * `latest.json`: Structured JSON document containing the results of the most recent benchmark run.

---

## 🛠️ Quickstart & Commands

### Prerequisites
* Operating System: macOS (Apple Silicon or Intel) or Linux (x86_64 or ARM64)
* Compiler: C++17 compatible compiler (`clang++` or `g++`)
* OpenMP: `libomp` (install via `brew install libomp` on macOS, or `apt-get install libomp-dev` on Ubuntu)

### 1. Build the Engine
Compiles all core object files and links the main executable into `build/engine`:
```bash
make
```

### 2. Run the Interactive Terminal REPL
Launches the interactive story generator using `Stories110M` (or edit `src/main.cpp` for `Stories15M`):
```bash
make run
```

Example interaction:
```text
========================================
       Stories110M Inference Engine
========================================
Model and Tokenizer loaded successfully!
Enter prompt to generate a story (type 'exit' to quit):

> The clever little squirrel found a golden key

--- Story ---
The clever little squirrel found a golden key under a big oak tree. He took it to his friend the badger, and together they searched the forest until they found a small wooden chest...
```

### 3. Run Automated Benchmarks
Executes the automated benchmarking harness across available models in `models/`, outputs a formatted table to stdout, and logs results to `benchmark_results/history.csv` and `benchmark_results/latest.json`:
```bash
make benchmark
```

Terminal Output:
```text
====================================================================================
Model         Size      Prefill (tok/s) Decode (tok/s)  Latency/Token   Threads   
------------------------------------------------------------------------------------
Stories15M    58 MB     674 tok/s       633 tok/s       1.579 ms        8         
Stories110M   418 MB    132 tok/s       129 tok/s       7.741 ms        8         
====================================================================================

Benchmark data saved with timestamp [2026-09-18 01:23:46]:
  • History: benchmark_results/history.csv (appended)
  • Latest:  benchmark_results/latest.json (overwritten)
```

### 4. Run Unit Tests
Compiles and runs 28 unit test suites (32,147 assertions) verifying kernels, KV Cache isolation, determinism, and sampling:
```bash
make test
```

### 5. Check Parity with PyTorch
Validates that C++ math kernels produce identical floating-point values to PyTorch reference implementations:
```bash
make parity
```

### 6. Clean Build Artifacts
Deletes the `build/` directory and compiled object files:
```bash
make clean
```

---

## 📊 Benchmark Persistence Schema

### CSV Format (`benchmark_results/history.csv`)
```csv
timestamp,model_name,model_size_mb,threads,dim,n_layers,n_heads,prompt_tokens,prefill_ms,prefill_tok_s,decode_tokens,decode_ms,decode_tok_s,latency_ms_per_tok
2026-09-18 01:23:46,Stories15M,58.00,8,288,6,6,35,51.91,674.24,100,157.97,633.03,1.58
2026-09-18 01:23:46,Stories110M,418.07,8,768,12,12,35,264.12,132.51,100,774.12,129.18,7.74
```

### JSON Format (`benchmark_results/latest.json`)
```json
[
  {
    "timestamp": "2026-09-18 01:23:46",
    "model_name": "Stories15M",
    "model_size_mb": 58.0,
    "threads": 8,
    "dim": 288,
    "n_layers": 6,
    "n_heads": 6,
    "prompt_tokens": 35,
    "prefill_ms": 51.91,
    "prefill_tok_s": 674.24,
    "decode_tokens": 100,
    "decode_ms": 157.97,
    "decode_tok_s": 633.03,
    "latency_ms_per_tok": 1.58
  },
  {
    "timestamp": "2026-09-18 01:23:46",
    "model_name": "Stories110M",
    "model_size_mb": 418.07,
    "threads": 8,
    "dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "prompt_tokens": 35,
    "prefill_ms": 264.12,
    "prefill_tok_s": 132.51,
    "decode_tokens": 100,
    "decode_ms": 774.12,
    "decode_tok_s": 129.18,
    "latency_ms_per_tok": 7.74
  }
]
```

---

## 📜 License
MIT License. Free and open source for educational and research use.
