#include "kv_cache.h"
#include <cstring>
#include <algorithm>

KVCache::KVCache(const Config& config)
    : n_layers(config.n_layers),
      seq_len(config.seq_len),
      n_heads(config.n_heads),
      n_kv_heads(config.n_kv_heads) {
    
    head_size = config.dim / config.n_heads;
    kv_dim = config.n_kv_heads * head_size;

    size_t total_elements = static_cast<size_t>(n_layers) * seq_len * kv_dim;
    key_cache.resize(total_elements, 0.0f);
    value_cache.resize(total_elements, 0.0f);
}

KVCache::KVCache(int n_layers, int seq_len, int dim, int n_heads, int n_kv_heads)
    : n_layers(n_layers),
      seq_len(seq_len),
      n_heads(n_heads),
      n_kv_heads(n_kv_heads) {
    
    head_size = dim / n_heads;
    kv_dim = n_kv_heads * head_size;

    size_t total_elements = static_cast<size_t>(n_layers) * seq_len * kv_dim;
    key_cache.resize(total_elements, 0.0f);
    value_cache.resize(total_elements, 0.0f);
}

float* KVCache::get_key(int layer, int pos) {
    return key_cache.data() + (layer * seq_len * kv_dim) + (pos * kv_dim);
}

const float* KVCache::get_key(int layer, int pos) const {
    return key_cache.data() + (layer * seq_len * kv_dim) + (pos * kv_dim);
}

float* KVCache::get_value(int layer, int pos) {
    return value_cache.data() + (layer * seq_len * kv_dim) + (pos * kv_dim);
}

const float* KVCache::get_value(int layer, int pos) const {
    return value_cache.data() + (layer * seq_len * kv_dim) + (pos * kv_dim);
}

float* KVCache::get_key_head(int layer, int pos, int kv_head) {
    return get_key(layer, pos) + (kv_head * head_size);
}

const float* KVCache::get_key_head(int layer, int pos, int kv_head) const {
    return get_key(layer, pos) + (kv_head * head_size);
}

float* KVCache::get_value_head(int layer, int pos, int kv_head) {
    return get_value(layer, pos) + (kv_head * head_size);
}

const float* KVCache::get_value_head(int layer, int pos, int kv_head) const {
    return get_value(layer, pos) + (kv_head * head_size);
}

void KVCache::store(int layer, int pos, const float* k, const float* v) {
    float* dst_k = get_key(layer, pos);
    float* dst_v = get_value(layer, pos);
    std::memcpy(dst_k, k, kv_dim * sizeof(float));
    std::memcpy(dst_v, v, kv_dim * sizeof(float));
}

void KVCache::clear() {
    std::fill(key_cache.begin(), key_cache.end(), 0.0f);
    std::fill(value_cache.begin(), value_cache.end(), 0.0f);
}

size_t KVCache::size_in_bytes() const {
    return (key_cache.size() + value_cache.size()) * sizeof(float);
}
