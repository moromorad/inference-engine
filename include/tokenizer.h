#pragma once

#include "model.h"
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// BYTE-PAIR ENCODING (BPE) TOKENIZER
// ============================================================================

/**
 * Byte-Pair Encoding (BPE) Tokenizer.
 *
 * Implements SentencePiece-compatible BPE tokenization for converting
 * raw text into integer token ID sequences, and vocabulary lookup for decoding.
 *
 * Execution stages during tokenization:
 *   1. Initial Character Encoding: Maps each character to its initial token ID.
 *   2. Greedy BPE Merging: Iteratively locates the adjacent pair of tokens with
 *      the highest merge score in `vocab_scores` and replaces them with their merged token ID.
 */
struct Tokenizer {
    // Array of string representations for each token indexed by token ID
    std::vector<std::string> vocab;

    // Merge priority scores for each token used by BPE greedy pairing
    std::vector<float> vocab_scores;

    // Hash map for O(1) string-to-token-ID lookup
    std::unordered_map<std::string, int> vocab_map;

    // Total number of tokens in the vocabulary
    int vocab_size;

    /**
     * Constructs a Tokenizer with an expected vocabulary size.
     *
     * @param size Expected number of vocabulary entries (e.g., 32000).
     */
    explicit Tokenizer(int size);

    /**
     * Loads vocabulary strings and merge scores from a binary tokenizer file.
     *
     * File binary structure:
     *   - 4 bytes: max_token_length (int)
     *   - For each token i in [0, vocab_size):
     *       - 4 bytes: float merge score
     *       - 4 bytes: int string length
     *       - length bytes: raw character string
     *
     * @param filepath Path to the binary tokenizer file (e.g., "models/tokenizer.bin").
     */
    void load(const char* filepath);

    /**
     * Tokenizes an input string into a vector of token IDs using BPE.
     *
     * @param text Raw input string.
     * @return     Vector of integer token IDs.
     */
    std::vector<int> tokenize(const std::string& text);

    /**
     * Searches the vocabulary hash map for a token string.
     *
     * @param text String piece to locate.
     * @return     Token ID if found, or -1 if not in vocabulary.
     */
    int find_token_id(const std::string& text);

    /**
     * Retrieves the embedding vector for a given token ID directly from the model.
     *
     * @param model    Reference to the loaded Model.
     * @param token_id Integer ID of the token.
     * @return         Pointer to the start of the token's embedding vector (size dim).
     */
    float* get_embedding(Model& model, int token_id);
};