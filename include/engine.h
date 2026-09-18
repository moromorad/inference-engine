#pragma once

#include "config.h"
#include "kv_cache.h"
#include "model.h"
#include "tokenizer.h"
#include <cstddef>
#include <string>
#include <vector>

// ============================================================================
// 1. RUNSTATE (SCRATCHPAD MEMORY)
// ============================================================================

/**
 * RunState: Pre-allocated scratchpad memory for single-token autoregressive decoding.
 *
 * Prevents dynamic heap allocations during the forward pass loop.
 * All intermediate activation buffers are sized at initialization based on Config
 * and reused across all Transformer layers and generation steps.
 */
struct RunState {
    // Current token activation / residual stream: size (dim)
    std::vector<float> x;

    // Normalized activation buffer: size (dim)
    std::vector<float> xb;

    // Secondary activation buffer (e.g., attention output before residual): size (dim)
    std::vector<float> xb2;

    // FFN hidden dimension branch 1 (W1 projection / SwiGLU output): size (hidden_dim)
    std::vector<float> hb;

    // FFN hidden dimension branch 2 (W3 projection): size (hidden_dim)
    std::vector<float> hb2;

    // Query projection buffer for current token: size (dim)
    std::vector<float> q;

    // Key projection buffer for current token: size (kv_dim = n_kv_heads * head_size)
    std::vector<float> k;

    // Value projection buffer for current token: size (kv_dim = n_kv_heads * head_size)
    std::vector<float> v;

    // Attention scores buffer across all sequence positions for all heads: size (n_heads * seq_len)
    std::vector<float> att;

    // Output logits over vocabulary: size (abs(vocab_size))
    std::vector<float> logits;

    /**
     * Allocate and size all scratchpad buffers according to the model configuration.
     *
     * @param config Model configuration containing dim, hidden_dim, n_heads, n_kv_heads, seq_len, vocab_size.
     */
    explicit RunState(const Config& config);
};


// ============================================================================
// 2. FORWARD PASS
// ============================================================================

/**
 * Executes a single Transformer Block (layer) pass.
 *
 * Execution stages:
 *   1. Attention Pre-Norm:     rmsnorm(state.xb, state.x, layer_rms_att, dim)
 *   2. Q, K, V Projections:    matmul into state.q, state.k, state.v
 *   3. RoPE:                   apply_rope on state.q and state.k at sequence index `pos`
 *   4. KV Cache Update:        kv_cache.store(layer, pos, state.k, state.v)
 *   5. Multi-Head Attention:   Scaled dot-product attention + softmax + weighted sum of cached values
 *   6. Attention Out Project:  matmul(state.xb2, concatenated_heads, layer_wo)
 *   7. Residual Connection 1:  accum(state.x, state.xb2, dim)
 *   8. FFN Pre-Norm:           rmsnorm(state.xb, state.x, layer_rms_ffn, dim)
 *   9. SwiGLU Up-Projections:  matmul(state.hb, state.xb, layer_w1), matmul(state.hb2, state.xb, layer_w3)
 *  10. SwiGLU Activation:      swiglu(state.hb, state.hb, state.hb2, hidden_dim)
 *  11. FFN Down-Projection:    matmul(state.xb, state.hb, layer_w2)
 *  12. Residual Connection 2:  accum(state.x, state.xb, dim)
 *
 * @param layer     Index of the Transformer block [0, config.n_layers).
 * @param pos       Current token position in sequence [0, config.seq_len).
 * @param model     Model containing architecture Config and weight matrices.
 * @param kv_cache  Key-Value cache holding historical keys and values.
 * @param state     Scratchpad buffers for intermediate activations.
 */
void transformer_block(int layer, int pos, const Model& model, KVCache& kv_cache, RunState& state);

/**
 * Executes the complete end-to-end forward pass for a single token at position `pos`.
 *
 * Execution flow:
 *   1. Embedding Lookup: Copies token embedding from model.weights.token_embedding_table into state.x
 *   2. Sequential Layers: Loops layer from 0 to config.n_layers - 1, calling transformer_block(...)
 *   3. Final RMSNorm: Normalizes state.x using model.weights.rms_final_weight
 *   4. Classifier (Logits): Projects normalized state.x to vocabulary logits using model.weights.wcls
 *
 * @param token     ID of current token being evaluated.
 * @param pos       Current sequence position index [0, config.seq_len).
 * @param model     Read-only Model containing weights and config.
 * @param kv_cache  Key-Value cache updated and read during attention.
 * @param state     Scratchpad memory used for intermediate activations.
 * @return          Pointer to the beginning of the logits vector (size vocab_size).
 */
float* forward(int token, int pos, const Model& model, KVCache& kv_cache, RunState& state);


// ============================================================================
// 3. SAMPLING METHODS
// ============================================================================

/**
 * Candidate token with its corresponding score/probability.
 * Used during Top-K sorting and selection.
 */
struct ProbIndex {
    float prob;
    int index;
};

/**
 * Greedy Argmax Sampling (used when temperature == 0.0f).
 * Scans the logits array and deterministically returns the index of the highest score.
 *
 * @param logits Array of raw prediction scores or probabilities.
 * @param size   Number of elements in the array (e.g., vocab_size).
 * @return       Index of the token with the highest score.
 */
int sample_argmax(const float* logits, int size);

/**
 * Top-K Sampling with Temperature Scaling (used when temperature > 0.0f).
 *
 * Execution flow:
 *   1. Scale:       Divides logits by temperature: scaled_logit = logit / temperature
 *   2. Top-K:       Selects the top K candidate tokens with the highest scaled logits
 *   3. Softmax:     Applies softmax across the top K candidates to form a probability distribution
 *   4. Sample:      Draws a candidate based on a random value in [0.0, 1.0)
 *
 * @param logits      Array of raw prediction scores (vocab_size elements).
 * @param vocab_size  Total size of the vocabulary.
 * @param k           Number of highest-probability candidate tokens to retain (e.g., 40).
 * @param temperature Temperature scaling factor (> 0.0f).
 * @param coin_flip   Random float in the range [0.0, 1.0). If negative, a random float will be generated.
 * @return            Selected token ID.
 */
int sample_top_k(float* logits, int vocab_size, int k, float temperature, float coin_flip = -1.0f);

/**
 * Unified Sampling Dispatcher.
 * Automatically delegates to:
 *   - sample_argmax: if temperature <= 0.0f
 *   - sample_top_k:  if temperature > 0.0f
 *
 * @param logits      Array of raw prediction scores (vocab_size elements).
 * @param vocab_size  Total size of the vocabulary.
 * @param temperature Sampling temperature (default: 1.0f, 0.0f for greedy argmax).
 * @param top_k       Top-K cutoff for candidate pool (default: 40).
 * @param coin_flip   Random value in [0.0, 1.0) for deterministic testing.
 * @return            Selected token ID.
 */
int sample(float* logits, int vocab_size, float temperature = 1.0f, int top_k = 40, float coin_flip = -1.0f);


// ============================================================================
// 4. AUTOREGRESSIVE GENERATION
// ============================================================================

/**
 * Hyperparameters and configuration for text generation.
 */
struct GenerationConfig {
    int max_new_tokens = 64;   // Maximum new tokens to generate
    float temperature = 0.8f;  // Sampling temperature (0.0f = greedy argmax)
    int top_k = 40;            // Top-K candidate pool cutoff
    int bos_token = 1;         // Beginning-of-Sequence token ID
    int eos_token = 2;         // End-of-Sequence token ID
};

/**
 * Autoregressive Generation Loop.
 *
 * Ingests prompt tokens to prefill the KV cache, then iteratively samples and
 * feeds back new tokens until max_new_tokens is reached, an EOS token is produced,
 * or the sequence length is exhausted.
 *
 * @param prompt     Initial user prompt string.
 * @param model      Loaded Model containing weights and architecture config.
 * @param tokenizer  Loaded Tokenizer for encoding prompt and decoding tokens.
 * @param gen_config Sampling parameters and token limits.
 * @param stream     If true, streams generated tokens to std::cout in real-time.
 * @return           Full generated text string.
 */
std::string generate(
    const std::string& prompt,
    const Model& model,
    Tokenizer& tokenizer,
    const GenerationConfig& gen_config = GenerationConfig(),
    bool stream = true
);
