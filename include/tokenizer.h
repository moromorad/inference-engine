#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "model.h"

struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::unordered_map<std::string, int> vocab_map;
    int vocab_size;

    // Constructor
    Tokenizer(int size);

    // Load function declaration
    void load(const char* filepath);

    // Tokenization (String -> array of token IDs)
    std::vector<int> tokenize(const std::string& text);

    // Finds token ID or returns -1
    int find_token_id(const std::string& text);

    // Returns embedding vector with Token ID token_id
    float* get_embedding(Model& model, int token_id);
};