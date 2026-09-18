#include "engine.h"
#include "kernels.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>

// ============================================================================
// RUNSTATE CONSTRUCTOR
// ============================================================================

RunState::RunState(const Config& config) {
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_size = dim / n_heads;
    int kv_dim = n_kv_heads * head_size;
    int seq_len = config.seq_len;
    int vocab_size = std::abs(config.vocab_size);

    x.resize(dim, 0.0f);
    xb.resize(dim, 0.0f);
    xb2.resize(dim, 0.0f);
    hb.resize(hidden_dim, 0.0f);
    hb2.resize(hidden_dim, 0.0f);
    q.resize(dim, 0.0f);
    k.resize(kv_dim, 0.0f);
    v.resize(kv_dim, 0.0f);
    att.resize(static_cast<size_t>(n_heads) * seq_len, 0.0f);
    logits.resize(vocab_size, 0.0f);
}

// ============================================================================
// TRANSFORMER BLOCK EXECUTION
// ============================================================================

void transformer_block(int layer, int pos, const Model& model, KVCache& kv_cache, RunState& state) {
    const Config& config = model.config;
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_size = dim / n_heads;
    int kv_dim = n_kv_heads * head_size;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = config.seq_len;

    // 0. Compute layer pointer offsets
    const float* rms_att = model.weights.rms_att_weight + (layer * dim);
    const float* wq = model.weights.wq + (layer * dim * dim);
    const float* wk = model.weights.wk + (layer * dim * kv_dim);
    const float* wv = model.weights.wv + (layer * dim * kv_dim);
    const float* wo = model.weights.wo + (layer * dim * dim);

    const float* rms_ffn = model.weights.rms_ffn_weight + (layer * dim);
    const float* w1 = model.weights.w1 + (layer * dim * hidden_dim);
    const float* w2 = model.weights.w2 + (layer * hidden_dim * dim);
    const float* w3 = model.weights.w3 + (layer * dim * hidden_dim);

    // ------------------------------------------------------------------------
    // PHASE 1: MULTI-HEAD ATTENTION SUB-LAYER
    // ------------------------------------------------------------------------

    // 1. Pre-Attention Normalization
    rmsnorm(state.xb.data(), state.x.data(), rms_att, dim);

    // 2. Q, K, V Projections (GEMV)
    matmul(state.q.data(), state.xb.data(), wq, dim, dim);
    matmul(state.k.data(), state.xb.data(), wk, dim, kv_dim);
    matmul(state.v.data(), state.xb.data(), wv, dim, kv_dim);

    // 3. Rotary Positional Embeddings (RoPE)
    apply_rope(state.q.data(), state.k.data(), pos, head_size, n_heads, n_kv_heads);

    // 4. Store current key and value into the KV Cache
    kv_cache.store(layer, pos, state.k.data(), state.v.data());

    // 5. Multi-Head Attention calculation across all heads
    #pragma omp parallel for
    for (int h = 0; h < n_heads; ++h) {
        int kv_head = h / kv_mul;
        const float* q_h = state.q.data() + (h * head_size);
        float* att_h = state.att.data() + (h * seq_len);

        // 5a. Scaled dot-product attention scores for all timesteps up to pos
        for (int t = 0; t <= pos; ++t) {
            const float* k_cached = kv_cache.get_key_head(layer, t, kv_head);
            float score = 0.0f;

            #pragma omp simd reduction(+:score)
            for (int i = 0; i < head_size; ++i) {
                score += q_h[i] * k_cached[i];
            }
            score /= std::sqrt(static_cast<float>(head_size));
            att_h[t] = score;
        }

        // 5b. Softmax over scores for positions [0, pos]
        softmax(att_h, pos + 1);

        // 5c. Weighted sum of cached values directly into output slice of state.xb
        float* out_head = state.xb.data() + (h * head_size);
        std::fill(out_head, out_head + head_size, 0.0f);

        for (int t = 0; t <= pos; ++t) {
            const float* v_cached = kv_cache.get_value_head(layer, t, kv_head);
            float weight = att_h[t];

            #pragma omp simd
            for (int i = 0; i < head_size; ++i) {
                out_head[i] += weight * v_cached[i];
            }
        }
    }

    // 6. Attention Output Projection (Wo) into state.xb2
    matmul(state.xb2.data(), state.xb.data(), wo, dim, dim);

    // 7. First Residual Connection: x = x + att_out
    accum(state.x.data(), state.xb2.data(), dim);

    // ------------------------------------------------------------------------
    // PHASE 2: FEED-FORWARD NETWORK SUB-LAYER (SwiGLU)
    // ------------------------------------------------------------------------

    // 8. Pre-FFN Normalization
    rmsnorm(state.xb.data(), state.x.data(), rms_ffn, dim);

    // 9. Up-Projections into W1 (Gate branch) and W3 (Value branch)
    matmul(state.hb.data(), state.xb.data(), w1, dim, hidden_dim);
    matmul(state.hb2.data(), state.xb.data(), w3, dim, hidden_dim);

    // 10. SwiGLU Activation: hb = SiLU(hb) * hb2
    swiglu(state.hb.data(), state.hb.data(), state.hb2.data(), hidden_dim);

    // 11. Down-Projection with W2 back to dim
    matmul(state.xb.data(), state.hb.data(), w2, hidden_dim, dim);

    // 12. Second Residual Connection: x = x + ffn_out
    accum(state.x.data(), state.xb.data(), dim);
}

// ============================================================================
// FULL FORWARD PASS EXECUTION
// ============================================================================

float* forward(int token, int pos, const Model& model, KVCache& kv_cache, RunState& state) {
    const Config& config = model.config;
    int dim = config.dim;
    int vocab_size = std::abs(config.vocab_size);

    // 1. Embedding lookup: copy embedding for token into state.x
    const float* token_emb = model.weights.token_embedding_table + (token * dim);
    std::memcpy(state.x.data(), token_emb, dim * sizeof(float));

    // 2. Sequentially execute all Transformer blocks
    for (int layer = 0; layer < config.n_layers; ++layer) {
        transformer_block(layer, pos, model, kv_cache, state);
    }

    // 3. Final RMSNorm
    rmsnorm(state.x.data(), state.x.data(), model.weights.rms_final_weight, dim);

    // 4. Classifier projection to logits (wcls)
    matmul(state.logits.data(), state.x.data(), model.weights.wcls, dim, vocab_size);

    return state.logits.data();
}

// ============================================================================
// SAMPLING IMPLEMENTATIONS
// ============================================================================

int sample_argmax(const float* logits, int size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int sample_top_k(float* logits, int vocab_size, int k, float temperature, float coin_flip) {
    if (k <= 0) k = 1;
    if (k > vocab_size) k = vocab_size;

    // 1. Scale logits by temperature
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    // 2. Collect candidates and find top-k using partial sort
    std::vector<ProbIndex> candidates(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        candidates[i] = {logits[i], i};
    }

    std::partial_sort(
        candidates.begin(),
        candidates.begin() + k,
        candidates.end(),
        [](const ProbIndex& a, const ProbIndex& b) {
            return a.prob > b.prob;
        }
    );
    candidates.resize(k);

    // 3. Numerically stable softmax over top-k candidates
    float max_val = candidates[0].prob; // Max is at index 0 due to descending sort
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        candidates[i].prob = std::exp(candidates[i].prob - max_val);
        sum += candidates[i].prob;
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < k; ++i) {
        candidates[i].prob *= inv_sum;
    }

    // 4. Sample token using coin_flip threshold
    float r = coin_flip;
    if (r < 0.0f || r >= 1.0f) {
        r = static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    }

    float cumulative = 0.0f;
    for (int i = 0; i < k; ++i) {
        cumulative += candidates[i].prob;
        if (r < cumulative) {
            return candidates[i].index;
        }
    }

    // Fallback to highest probability candidate
    return candidates[0].index;
}

int sample(float* logits, int vocab_size, float temperature, int top_k, float coin_flip) {
    if (temperature <= 0.0f) {
        return sample_argmax(logits, vocab_size);
    }
    return sample_top_k(logits, vocab_size, top_k, temperature, coin_flip);
}

// ============================================================================
// AUTOREGRESSIVE GENERATION IMPLEMENTATION
// ============================================================================

std::string generate(
    const std::string& prompt,
    const Model& model,
    Tokenizer& tokenizer,
    const GenerationConfig& gen_config,
    bool stream
) {
    const Config& config = model.config;
    int vocab_size = std::abs(config.vocab_size);

    KVCache kv_cache(config);
    RunState state(config);

    // 1. Tokenize prompt
    std::vector<int> prompt_tokens;
    if (!prompt.empty()) {
        prompt_tokens = tokenizer.tokenize(prompt);
    }

    // Prepend BOS token (token 1) if not present
    if (prompt_tokens.empty() || prompt_tokens[0] != gen_config.bos_token) {
        prompt_tokens.insert(prompt_tokens.begin(), gen_config.bos_token);
    }

    std::string full_output = "";

    // 2. Prefill Phase: Ingest all prompt tokens except the last one
    int num_prompt_tokens = static_cast<int>(prompt_tokens.size());
    for (int i = 0; i < num_prompt_tokens - 1; ++i) {
        forward(prompt_tokens[i], i, model, kv_cache, state);
        if (i > 0) { // Skip printing the special BOS token
            full_output += tokenizer.vocab[prompt_tokens[i]];
            if (stream) {
                std::cout << tokenizer.vocab[prompt_tokens[i]];
                std::cout.flush();
            }
        }
    }

    // Ingest the last prompt token and obtain logits for the first new token
    int last_prompt_idx = num_prompt_tokens - 1;
    float* logits = forward(prompt_tokens[last_prompt_idx], last_prompt_idx, model, kv_cache, state);
    if (last_prompt_idx > 0) {
        full_output += tokenizer.vocab[prompt_tokens[last_prompt_idx]];
        if (stream) {
            std::cout << tokenizer.vocab[prompt_tokens[last_prompt_idx]];
            std::cout.flush();
        }
    }

    int next_token = sample(logits, vocab_size, gen_config.temperature, gen_config.top_k);

    // 3. Decoding Phase: Autoregressive feedback loop
    int pos = num_prompt_tokens;
    int generated_count = 0;
    int current_token = next_token;

    while (pos < config.seq_len && generated_count < gen_config.max_new_tokens) {
        // Stop if EOS is encountered
        if (current_token == gen_config.eos_token || current_token == gen_config.bos_token) {
            break;
        }

        std::string piece = tokenizer.vocab[current_token];
        full_output += piece;
        if (stream) {
            std::cout << piece;
            std::cout.flush();
        }

        // Forward pass for the newly generated token
        logits = forward(current_token, pos, model, kv_cache, state);

        // Sample the next token from the newly computed logits
        current_token = sample(logits, vocab_size, gen_config.temperature, gen_config.top_k);

        pos++;
        generated_count++;
    }

    if (stream) {
        std::cout << std::endl;
    }

    return full_output;
}

