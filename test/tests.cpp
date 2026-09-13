#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../include/doctest.h"
#include "../include/tokenizer.h"
#include "../include/model.h"
#include "../include/kernels.h"
#include "../include/kv_cache.h"
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

TEST_SUITE("Matmul Kernel") {
    TEST_CASE("Matmul - Hand-calculated 3x2 matrix-vector multiplication") {
        // x: shape (1, 3)
        // w: shape (2, 3) -> 2 output rows, 3 input columns
        // out: shape (1, 2)
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> w = {
            1.0f, 2.0f, 3.0f,  // Row 0: 1*1 + 2*2 + 3*3 = 14
            4.0f, 5.0f, 6.0f   // Row 1: 1*4 + 2*5 + 3*6 = 32
        };
        std::vector<float> out(2, 0.0f);

        matmul(out.data(), x.data(), w.data(), 3, 2);

        CHECK(out[0] == doctest::Approx(14.0f));
        CHECK(out[1] == doctest::Approx(32.0f));
    }

    TEST_CASE("Matmul - 4x4 Identity Matrix preserves input vector") {
        constexpr int dim = 4;
        std::vector<float> x = {0.5f, -1.2f, 3.0f, 4.5f};
        std::vector<float> identity_w(dim * dim, 0.0f);
        for (int i = 0; i < dim; ++i) {
            identity_w[i * dim + i] = 1.0f;
        }
        std::vector<float> out(dim, 0.0f);

        matmul(out.data(), x.data(), identity_w.data(), dim, dim);

        for (int i = 0; i < dim; ++i) {
            CHECK(out[i] == doctest::Approx(x[i]));
        }
    }
}

TEST_SUITE("Softmax Kernel") {
    TEST_CASE("Softmax - Equal inputs produce equal probabilities") {
        std::vector<float> x = {2.0f, 2.0f};
        softmax(x.data(), 2);

        CHECK(x[0] == doctest::Approx(0.5f));
        CHECK(x[1] == doctest::Approx(0.5f));
        CHECK((x[0] + x[1]) == doctest::Approx(1.0f));
    }

    TEST_CASE("Softmax - Numerical stability with large logits (no overflow)") {
        // Large values that would overflow exp() without max subtraction
        std::vector<float> x = {1000.0f, 1001.0f, 1002.0f};
        softmax(x.data(), 3);

        // Expected values:
        // shifted = [-2.0, -1.0, 0.0]
        // exp = [0.135335, 0.367879, 1.0] -> sum = 1.503214
        CHECK(x[0] == doctest::Approx(0.09003057f).epsilon(1e-4f));
        CHECK(x[1] == doctest::Approx(0.24472847f).epsilon(1e-4f));
        CHECK(x[2] == doctest::Approx(0.66524096f).epsilon(1e-4f));

        float total = x[0] + x[1] + x[2];
        CHECK(total == doctest::Approx(1.0f));
    }
}

TEST_SUITE("SiLU Kernel") {
    TEST_CASE("SiLU - Hand-calculated values") {
        // x * sigmoid(x)
        std::vector<float> x = {0.0f, 2.0f, -2.0f};
        silu(x.data(), 3);

        // silu(0) = 0 * 0.5 = 0.0
        CHECK(x[0] == doctest::Approx(0.0f));
        // silu(2) = 2 / (1 + exp(-2)) ≈ 1.761594f
        CHECK(x[1] == doctest::Approx(1.761594f).epsilon(1e-4f));
        // silu(-2) = -2 / (1 + exp(2)) ≈ -0.238406f
        CHECK(x[2] == doctest::Approx(-0.238406f).epsilon(1e-4f));
    }
}

TEST_SUITE("SwiGLU Kernel") {
    TEST_CASE("SwiGLU - Computes SiLU(hb1) * hb2 element-wise") {
        std::vector<float> hb1 = {0.0f, 2.0f};
        std::vector<float> hb2 = {5.0f, 3.0f};
        std::vector<float> out(2, 0.0f);

        swiglu(out.data(), hb1.data(), hb2.data(), 2);

        // out[0] = silu(0.0) * 5.0 = 0.0 * 5.0 = 0.0
        CHECK(out[0] == doctest::Approx(0.0f));
        // out[1] = silu(2.0) * 3.0 ≈ 1.761594 * 3.0 ≈ 5.284782f
        CHECK(out[1] == doctest::Approx(5.284782f).epsilon(1e-4f));
    }
}

TEST_SUITE("Accum Kernel") {
    TEST_CASE("Accum - In-place element-wise addition (Residual Connection)") {
        std::vector<float> a = {1.5f, -2.0f, 0.0f, 4.2f};
        std::vector<float> b = {0.5f,  3.0f, -1.0f, -4.2f};

        accum(a.data(), b.data(), 4);

        CHECK(a[0] == doctest::Approx(2.0f));
        CHECK(a[1] == doctest::Approx(1.0f));
        CHECK(a[2] == doctest::Approx(-1.0f));
        CHECK(a[3] == doctest::Approx(0.0f));
    }
}

TEST_SUITE("RoPE Kernel") {
    TEST_CASE("RoPE - Position 0 causes zero rotation (identity)") {
        std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> k = {5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> q_orig = q;
        std::vector<float> k_orig = k;

        apply_rope(q.data(), k.data(), 0, 4, 1, 1);

        for (int i = 0; i < 4; ++i) {
            CHECK(q[i] == doctest::Approx(q_orig[i]));
            CHECK(k[i] == doctest::Approx(k_orig[i]));
        }
    }

    TEST_CASE("RoPE - Analytical 2D rotation and length preservation") {
        // head_size = 2, pos = 1 -> freq = 1.0, theta = 1.0 rad
        // cos(1.0) ≈ 0.5403023, sin(1.0) ≈ 0.84147098
        std::vector<float> q = {1.0f, 0.0f};
        std::vector<float> k = {0.0f, 1.0f};

        apply_rope(q.data(), k.data(), 1, 2, 1, 1);

        // q: [1, 0] rotated by 1.0 rad -> [cos(1), sin(1)]
        CHECK(q[0] == doctest::Approx(std::cos(1.0f)));
        CHECK(q[1] == doctest::Approx(std::sin(1.0f)));

        // k: [0, 1] rotated by 1.0 rad -> [-sin(1), cos(1)]
        CHECK(k[0] == doctest::Approx(-std::sin(1.0f)));
        CHECK(k[1] == doctest::Approx(std::cos(1.0f)));

        // Length preservation: rotation must not alter the vector length
        float q_norm = std::sqrt(q[0] * q[0] + q[1] * q[1]);
        float k_norm = std::sqrt(k[0] * k[0] + k[1] * k[1]);
        CHECK(q_norm == doctest::Approx(1.0f));
        CHECK(k_norm == doctest::Approx(1.0f));
    }

    TEST_CASE("RoPE - Multi-head offsets operate correctly across heads") {
        // 2 Query heads of head_size = 2 -> total 4 elements in q
        // 1 Key head of head_size = 2   -> total 2 elements in k
        std::vector<float> q = {1.0f, 0.0f,  1.0f, 0.0f};
        std::vector<float> k = {0.0f, 1.0f};

        apply_rope(q.data(), k.data(), 1, 2, 2, 1);

        // Both Q heads should be rotated identically
        CHECK(q[0] == doctest::Approx(std::cos(1.0f)));
        CHECK(q[1] == doctest::Approx(std::sin(1.0f)));
        CHECK(q[2] == doctest::Approx(std::cos(1.0f)));
        CHECK(q[3] == doctest::Approx(std::sin(1.0f)));

        CHECK(k[0] == doctest::Approx(-std::sin(1.0f)));
        CHECK(k[1] == doctest::Approx(std::cos(1.0f)));
    }
}

TEST_SUITE("KVCache") {
    TEST_CASE("KVCache - Initialization, Dimensions, and Memory Sizing") {
        // Architecture: dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, seq_len=256
        Config config = {288, 768, 6, 6, 6, 32000, 256};
        KVCache cache(config);

        CHECK(cache.n_layers == 6);
        CHECK(cache.seq_len == 256);
        CHECK(cache.n_heads == 6);
        CHECK(cache.n_kv_heads == 6);
        CHECK(cache.head_size == 48); // 288 / 6 = 48
        CHECK(cache.kv_dim == 288);    // 6 * 48 = 288

        size_t expected_elements = 6 * 256 * 288;
        CHECK(cache.key_cache.size() == expected_elements);
        CHECK(cache.value_cache.size() == expected_elements);
        CHECK(cache.size_in_bytes() == 2 * expected_elements * sizeof(float));
    }

    TEST_CASE("KVCache - Store, Retrieve, and Layer/Position Isolation") {
        // 2 layers, max sequence 4, dim 4, 2 heads, 2 kv_heads (head_size = 2, kv_dim = 4)
        KVCache cache(2, 4, 4, 2, 2);

        std::vector<float> k_layer0_pos0 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> v_layer0_pos0 = {5.0f, 6.0f, 7.0f, 8.0f};

        std::vector<float> k_layer0_pos1 = {10.0f, 20.0f, 30.0f, 40.0f};
        std::vector<float> v_layer0_pos1 = {50.0f, 60.0f, 70.0f, 80.0f};

        std::vector<float> k_layer1_pos0 = {-1.0f, -2.0f, -3.0f, -4.0f};
        std::vector<float> v_layer1_pos0 = {-5.0f, -6.0f, -7.0f, -8.0f};

        cache.store(0, 0, k_layer0_pos0.data(), v_layer0_pos0.data());
        cache.store(0, 1, k_layer0_pos1.data(), v_layer0_pos1.data());
        cache.store(1, 0, k_layer1_pos0.data(), v_layer1_pos0.data());

        // Verify layer 0, pos 0
        float* retrieved_k = cache.get_key(0, 0);
        float* retrieved_v = cache.get_value(0, 0);
        for (int i = 0; i < 4; ++i) {
            CHECK(retrieved_k[i] == doctest::Approx(k_layer0_pos0[i]));
            CHECK(retrieved_v[i] == doctest::Approx(v_layer0_pos0[i]));
        }

        // Verify layer 0, pos 1
        retrieved_k = cache.get_key(0, 1);
        retrieved_v = cache.get_value(0, 1);
        for (int i = 0; i < 4; ++i) {
            CHECK(retrieved_k[i] == doctest::Approx(k_layer0_pos1[i]));
            CHECK(retrieved_v[i] == doctest::Approx(v_layer0_pos1[i]));
        }

        // Verify layer 1, pos 0
        retrieved_k = cache.get_key(1, 0);
        retrieved_v = cache.get_value(1, 0);
        for (int i = 0; i < 4; ++i) {
            CHECK(retrieved_k[i] == doctest::Approx(k_layer1_pos0[i]));
            CHECK(retrieved_v[i] == doctest::Approx(v_layer1_pos0[i]));
        }

        // Verify head-level access (head 1 of layer 0, pos 1: elements [30.0, 40.0])
        float* head1_k = cache.get_key_head(0, 1, 1);
        CHECK(head1_k[0] == doctest::Approx(30.0f));
        CHECK(head1_k[1] == doctest::Approx(40.0f));
    }
}