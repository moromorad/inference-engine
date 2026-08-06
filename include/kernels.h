#pragma once
#include <cstddef>

// ============================================================================
// 1. NORMALIZATION & ACTIVATION KERNELS
// ============================================================================

/**
 * Root Mean Square Normalization (RMSNorm).
 * Normalizes input vector x in-place or into out, scaled by weight vector.
 *
 * @param out    Destination array (size elements).
 * @param x      Source array (size elements).
 * @param weight Learned scale parameters (size elements).
 * @param size   Dimension size (e.g., dim = 288).
 * @param eps    Small constant to prevent division by zero (default: 1e-5f).
 */
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps = 1e-5f);

/**
 * Numerically Stable Softmax.
 * Subtracts max value before exponentiating to prevent float32 overflow.
 * Operates in-place on vector x.
 *
 * @param x    Array of raw scores/logits (modified in-place to probabilities).
 * @param size Number of elements (e.g., vocab_size or sequence length).
 */
void softmax(float* x, int size);

/**
 * SiLU (Sigmoid Linear Unit) activation: f(x) = x * sigmoid(x)
 * Applied element-wise in-place.
 *
 * @param x    Array to activate in-place.
 * @param size Number of elements.
 */
void silu(float* x, int size);

/**
 * SwiGLU Feed-Forward activation pass.
 * Computes: out = SiLU(hb1) * hb2
 *
 * @param out        Destination array (hidden_dim elements).
 * @param hb1        First branch output (W1 * x).
 * @param hb2        Gated branch output (W3 * x).
 * @param hidden_dim Dimension size of the FFN hidden layer.
 */
void swiglu(float* out, const float* hb1, const float* hb2, int hidden_dim);


// ============================================================================
// 2. LINEAR ALGEBRA & TRANSFORMATIONS (GEMV)
// ============================================================================

/**
 * Vector-Matrix Multiplication (GEMV).
 * Computes: out = x * W
 * 
 * Note: Single-token inference reduces general GEMM (Matrix-Matrix) to GEMV.
 *
 * @param out Destination vector of size d.
 * @param x   Input vector of size n.
 * @param w   Weight matrix stored as a contiguous 1D array of shape (n x d).
 * @param n   Input dimension.
 * @param d   Output dimension.
 */
void matmul(float* out, const float* x, const float* w, int n, int d);

/**
 * Element-wise Vector Accumulation (Residual Connection).
 * Computes: a = a + b
 *
 * @param a    Destination and first operand array (size elements).
 * @param b    Second operand array (size elements).
 * @param size Dimension size.
 */
void accum(float* a, const float* b, int size);


// ============================================================================
// 3. POSITIONAL EMBEDDINGS & ATTENTION KERNELS
// ============================================================================

/**
 * Rotary Positional Embeddings (RoPE).
 * Rotates pairs of dimensions in Query and Key vectors according to token position.
 * Operates in-place on q and k.
 *
 * @param q          Query vector for current token (dim elements).
 * @param k          Key vector for current token (kv_dim elements).
 * @param pos        Absolute position index of current token in sequence.
 * @param head_size  Dimension per head (e.g., dim / n_heads).
 * @param n_heads    Number of Query heads.
 * @param n_kv_heads Number of Key/Value heads.
 */
void apply_rope(float* q, float* k, int pos, int head_size, int n_heads, int n_kv_heads);

