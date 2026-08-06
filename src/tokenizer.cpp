
#include "tokenizer.h"
#include <iostream>



// Constructor to set the size
Tokenizer::Tokenizer(int size) : vocab_size(size) {}

// The load function we just discussed
void Tokenizer::load(const char* filepath) {
    FILE* file = fopen(filepath, "rb");

    if (!file) {
        std::cerr << "Failed to open " << filepath << "\n";
        exit(1);
    }

    int max_token_length;
    if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
        std::cerr << "Failed to read header\n";
        exit(1);
    }

    vocab.resize(vocab_size);
    vocab_scores.resize(vocab_size);
    char* word_buffer = new char[max_token_length];

    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        fread(&score, sizeof(float), 1, file);
        vocab_scores[i] = score;
            
        fread(&len, sizeof(int), 1, file);
        fread(word_buffer, len, 1, file);
        vocab[i] = std::string(word_buffer, len);
        vocab_map[vocab[i]] = i;
    }

    delete[] word_buffer;
    fclose(file);
    std::cout << "Tokenizer loaded successfully.\n";
}


int Tokenizer::find_token_id(const std::string& text) {
    auto it = vocab_map.find(text);
    if (it != vocab_map.end()) {
        return it->second; // Returns the ID
    }
    return -1; // Not found
}

float* Tokenizer::get_embedding(Model& model, int token_id) {
    return &model.weights.token_embedding_table[token_id * model.config.dim];
}

std::vector<int> Tokenizer::tokenize(const std::string& text) {
    std::vector<int> tokens;

    // Phase 1: Encode each character as its own token
    for (size_t i = 0; i < text.size(); i++) {
        std::string ch(1, text[i]);
        int id = find_token_id(ch);
        if (id == -1) {
            std::cerr << "Character not in vocab: '" << ch << "'\n";
            continue;
        }
        tokens.push_back(id);
    }

    // Phase 2: BPE merge loop — greedily merge the highest-scoring pair
    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            std::string merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
            int id = find_token_id(merged);
            if (id != -1 && vocab_scores[id] > best_score) {
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break; // No more merges possible

        // Replace the pair at best_idx with the merged token
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    return tokens;
}