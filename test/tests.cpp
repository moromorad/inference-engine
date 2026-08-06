#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../include/doctest.h"
#include "../include/tokenizer.h"
#include "../include/model.h"
#include "../include/kernels.h"
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

// Helper: set up a tokenizer with a custom vocabulary for BPE tests
// vocab entries:  0="a", 1="b", 2="c", 3="ab", 4="abc"
// scores chosen so that "ab" (score 10) merges before "abc" can be looked up,
// and then "ab"+"c" → "abc" (score 20) merges in the next round.
Tokenizer make_bpe_tokenizer() {
    Tokenizer t(5);
    t.vocab       = {"a", "b", "c", "ab", "abc"};
    t.vocab_scores = {0.0f, 0.0f, 0.0f, 10.0f, 20.0f};
    for (int i = 0; i < 5; i++) {
        t.vocab_map[t.vocab[i]] = i;
    }
    return t;
}

TEST_CASE("Tokenize: single characters, no merges possible") {
    Tokenizer t = make_bpe_tokenizer();

    // "ca" — no merge token exists for "ca", so stays as two char tokens
    auto tokens = t.tokenize("ca");
    REQUIRE(tokens.size() == 2);
    CHECK(tokens[0] == 2); // 'c'
    CHECK(tokens[1] == 0); // 'a'
}

TEST_CASE("Tokenize: single merge") {
    Tokenizer t = make_bpe_tokenizer();

    // "ab" → chars [a=0, b=1] → merge into "ab"=3
    auto tokens = t.tokenize("ab");
    REQUIRE(tokens.size() == 1);
    CHECK(tokens[0] == 3); // "ab"
}

TEST_CASE("Tokenize: multi-step BPE merge") {
    Tokenizer t = make_bpe_tokenizer();

    // "abc" → chars [a=0, b=1, c=2]
    //  round 1: best merge is a+b → "ab"=3 (score 10)  → [3, 2]
    //  round 2: best merge is ab+c → "abc"=4 (score 20) → [4]
    auto tokens = t.tokenize("abc");
    REQUIRE(tokens.size() == 1);
    CHECK(tokens[0] == 4); // "abc"
}

TEST_CASE("Tokenize: merge picks highest score first") {
    // vocab: 0="x", 1="y", 2="z", 3="xy" (score 5), 4="yz" (score 15)
    Tokenizer t(5);
    t.vocab        = {"x", "y", "z", "xy", "yz"};
    t.vocab_scores = {0.0f, 0.0f, 0.0f, 5.0f, 15.0f};
    for (int i = 0; i < 5; i++) t.vocab_map[t.vocab[i]] = i;

    // "xyz" → chars [x=0, y=1, z=2]
    // Two candidate merges: x+y→"xy"(5) and y+z→"yz"(15)
    // "yz" wins first → [0, 4]  then no merge for "x"+"yz" → done
    auto tokens = t.tokenize("xyz");
    REQUIRE(tokens.size() == 2);
    CHECK(tokens[0] == 0); // 'x'
    CHECK(tokens[1] == 4); // "yz"
}

TEST_CASE("Tokenize: empty string") {
    Tokenizer t = make_bpe_tokenizer();
    auto tokens = t.tokenize("");
    CHECK(tokens.empty());
}

TEST_SUITE("RMSNorm Kernel") {
    
    // 1. Basic Analytical Test (Hand-calculated, no PyTorch needed)
    TEST_CASE("RMSNorm - Hand-Calculated Sanity Check") {
        // Input: [2.0, 0.0, 0.0, 0.0], N = 4, eps = 0.0
        // Sum of squares = 4.0, Mean = 1.0, RMS = 1.0
        std::vector<float> x = {2.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> out(4, 0.0f);

        rmsnorm(out.data(), x.data(), weight.data(), 4, 0.0f);

        CHECK(out[0] == doctest::Approx(2.0f));
        CHECK(out[1] == doctest::Approx(0.0f));
        CHECK(out[2] == doctest::Approx(0.0f));
        CHECK(out[3] == doctest::Approx(0.0f));
    }

    TEST_CASE("RMSNorm - Numerical parity with PyTorch") {
        constexpr int size = 4;
        constexpr float eps = 1e-5f;

        // Stack allocation only (zero dynamic allocation)
        const float x[size] = {1.0f, -2.0f, 3.0f, -4.0f};
        const float weight[size] = {0.5f, 1.5f, 1.0f, 0.8f};
        float out[size] = {0.0f};

        // PyTorch golden values
        const float expected[size] = {
            0.18257406f,
            -1.09544444f,
            1.09544444f,
            -1.16847408f
        };

        rmsnorm(out, x, weight, size, eps);

        for (int i = 0; i < size; ++i) {
            CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-5f));
        }
    }
}