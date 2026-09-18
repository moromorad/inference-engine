#pragma once

#include "config.h"
#include <cstddef>
#include <vector>

// ============================================================================
// KEY-VALUE CACHE (KV CACHE)
// ============================================================================

/**
 * KV Cache (Key-Value Cache) for Transformer Autoregressive Generation.
 *
 * Pre-allocates and manages contiguous memory for Key and Value vectors
 * across all Transformer layers and token sequence positions.
 *
 * Logical 4D Shape:
 *   [n_layers][seq_len][n_kv_heads][head_size]
 *
 * Flattened 1D Memory Layout per cache buffer (key_cache and value_cache):
 *   Size = n_layers * seq_len * kv_dim
 *   where:
 *     head_size = dim / n_heads
 *     kv_dim    = n_kv_heads * head_size
 */
class KVCache {
public:
    int n_layers;
    int seq_len;
    int kv_dim;
    int head_size;
    int n_heads;
    int n_kv_heads;

    // Contiguous pre-allocated flat storage buffers (allocated ONCE at boot)
    std::vector<float> key_cache;
    std::vector<float> value_cache;

    /**
     * Construct KVCache using the model's architecture Config.
     *
     * @param config Model configuration containing dim, n_layers, n_heads, n_kv_heads, seq_len.
     */
    explicit KVCache(const Config& config);

    /**
     * Construct KVCache with explicit architecture parameters.
     */
    KVCache(int n_layers, int seq_len, int dim, int n_heads, int n_kv_heads);

    /**
     * Pointer to the start of the Key vector for Layer `layer` and Position `pos`.
     * The vector contains `kv_dim` contiguous floats.
     */
    float* get_key(int layer, int pos);
    const float* get_key(int layer, int pos) const;

    /**
     * Pointer to the start of the Value vector for Layer `layer` and Position `pos`.
     * The vector contains `kv_dim` contiguous floats.
     */
    float* get_value(int layer, int pos);
    const float* get_value(int layer, int pos) const;

    /**
     * Pointer to a specific Key head for Layer `layer`, Position `pos`, and Head `kv_head`.
     * The vector contains `head_size` contiguous floats.
     */
    float* get_key_head(int layer, int pos, int kv_head);
    const float* get_key_head(int layer, int pos, int kv_head) const;

    /**
     * Pointer to a specific Value head for Layer `layer`, Position `pos`, and Head `kv_head`.
     * The vector contains `head_size` contiguous floats.
     */
    float* get_value_head(int layer, int pos, int kv_head);
    const float* get_value_head(int layer, int pos, int kv_head) const;

    /**
     * Store newly computed key and value vectors into the cache at (layer, pos).
     *
     * @param layer Layer index [0, n_layers).
     * @param pos   Current token position [0, seq_len).
     * @param k     Source key array of size kv_dim (typically post-RoPE).
     * @param v     Source value array of size kv_dim.
     */
    void store(int layer, int pos, const float* k, const float* v);

    /**
     * Reset the entire cache to zero.
     */
    void clear();

    /**
     * Total memory allocated by both Key and Value buffers in bytes.
     */
    size_t size_in_bytes() const;
};
