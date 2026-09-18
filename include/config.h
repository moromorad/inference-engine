#pragma once

// ============================================================================
// 1. MODEL ARCHITECTURE CONFIGURATION
// ============================================================================

/**
 * Model Architecture Configuration.
 *
 * Directly mirrors the 28-byte binary header exported at the beginning
 * of the model weights file (e.g., stories15M.bin, stories110M.bin).
 */
struct Config {
    int dim;        // Model / token embedding dimension (d_model, e.g., 288 or 768)
    int hidden_dim; // Internal hidden dimension for the SwiGLU Feed-Forward Network (d_ff, e.g., 768 or 2048)
    int n_layers;   // Number of Transformer decoder blocks stacked sequentially (e.g., 6 or 12)
    int n_heads;    // Number of Query attention heads (e.g., 6 or 12)
    int n_kv_heads; // Number of Key/Value attention heads (enables Grouped-Query Attention when n_kv_heads < n_heads)
    int vocab_size; // Vocabulary size (positive: classifier shares embedding table; negative: unshared classifier)
    int seq_len;    // Maximum context / sequence length supported by the KV cache and positional embeddings (e.g., 256 or 1024)
};


// ============================================================================
// 2. TRANSFORMER WEIGHT POINTERS
// ============================================================================

/**
 * Pointers to Model Weight Matrices.
 *
 * These float pointers map directly into the memory-mapped (mmap) binary file,
 * enabling instant zero-copy loading without allocating duplicate RAM.
 *
 * Weight Layout across Layers:
 *   - Attention & FFN weights for all layers are stored contiguously in memory.
 *   - Pointers point to the beginning of Layer 0; subsequent layers are accessed via offsets.
 */
struct TransformerWeights {
    // Token embedding lookup table: shape (vocab_size, dim)
    float* token_embedding_table;

    // --- Attention Weights (stacked across all n_layers) ---
    // Pre-attention RMSNorm scale weights: shape (n_layers, dim)
    float* rms_att_weight;
    // Query projection weights: shape (n_layers, dim, n_heads * head_size)
    float* wq;
    // Key projection weights: shape (n_layers, dim, n_kv_heads * head_size)
    float* wk;
    // Value projection weights: shape (n_layers, dim, n_kv_heads * head_size)
    float* wv;
    // Output projection weights: shape (n_layers, n_heads * head_size, dim)
    float* wo;

    // --- Feed-Forward / SwiGLU Weights (stacked across all n_layers) ---
    // Pre-FFN RMSNorm scale weights: shape (n_layers, dim)
    float* rms_ffn_weight;
    // FFN Gate branch (W1) projection weights: shape (n_layers, dim, hidden_dim)
    float* w1;
    // FFN Down-projection (W2) weights: shape (n_layers, hidden_dim, dim)
    float* w2;
    // FFN Value branch (W3) projection weights: shape (n_layers, dim, hidden_dim)
    float* w3;

    // --- Final Normalization & Classifier ---
    // Final RMSNorm scale weight before output classifier: shape (dim)
    float* rms_final_weight;
    // Vocabulary output projection classifier (LM Head): shape (vocab_size, dim)
    // Points to token_embedding_table when weight tying is used.
    float* wcls;
};
