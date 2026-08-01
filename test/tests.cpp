#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../include/doctest.h"
#include "../include/tokenizer.h"
#include "../include/model.h"
#include <fstream>
#include <cstdio>
#include <vector>

TEST_CASE("Testing Tokenizer::find_token_id") {
    Tokenizer tokenizer(100);
    
    // Manually populate vocab_map for testing
    tokenizer.vocab_map["hello"] = 42;
    tokenizer.vocab_map["world"] = 99;

    CHECK(tokenizer.find_token_id("hello") == 42);
    CHECK(tokenizer.find_token_id("world") == 99);
    CHECK(tokenizer.find_token_id("unknown") == -1);
}

// Helper to create a dummy model for the embedding test
void create_dummy_model(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    // Config: dim=4, hidden_dim=4, n_layers=0, n_heads=1, n_kv_heads=1, vocab_size=2, seq_len=10
    // We set n_layers=0 to prevent the Model pointer walk from going out of bounds
    Config config = {4, 4, 0, 1, 1, 2, 10}; 
    out.write(reinterpret_cast<const char*>(&config), sizeof(Config));
    
    // Data for token embedding table (2 tokens * 4 dim = 8 floats) 
    // + rms_final_weight (4 floats)
    std::vector<float> data(12);
    
    // Token 0 embedding
    data[0] = 0.1f; data[1] = 0.2f; data[2] = 0.3f; data[3] = 0.4f;
    // Token 1 embedding
    data[4] = 0.5f; data[5] = 0.6f; data[6] = 0.7f; data[7] = 0.8f;
    // rms_final_weight
    data[8] = 1.0f; data[9] = 1.0f; data[10] = 1.0f; data[11] = 1.0f;
    
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close();
}

TEST_CASE("Testing Tokenizer::get_embedding") {
    std::string dummy_file = "test_dummy_model.bin";
    create_dummy_model(dummy_file);
    
    Model model(dummy_file);
    Tokenizer tokenizer(2); // vocab_size = 2
    
    float* emb0 = tokenizer.get_embedding(model, 0);
    CHECK(emb0[0] == doctest::Approx(0.1f));
    CHECK(emb0[1] == doctest::Approx(0.2f));
    CHECK(emb0[2] == doctest::Approx(0.3f));
    CHECK(emb0[3] == doctest::Approx(0.4f));
    
    float* emb1 = tokenizer.get_embedding(model, 1);
    CHECK(emb1[0] == doctest::Approx(0.5f));
    CHECK(emb1[1] == doctest::Approx(0.6f));
    CHECK(emb1[2] == doctest::Approx(0.7f));
    CHECK(emb1[3] == doctest::Approx(0.8f));
    
    // Cleanup temporary file
    std::remove(dummy_file.c_str());
}
