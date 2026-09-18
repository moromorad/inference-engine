#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <cmath>
#include <sys/stat.h>
#include <omp.h>
#include "model.h"
#include "tokenizer.h"
#include "engine.h"
#include "kv_cache.h"

// Struct to hold benchmark results for a single model run
struct BenchmarkResult {
    std::string timestamp;
    std::string model_name;
    double model_size_mb;
    int num_threads;
    int dim;
    int n_layers;
    int n_heads;
    int prompt_tokens;
    double prefill_ms;
    double prefill_tok_s;
    int decode_tokens;
    double decode_ms;
    double decode_tok_s;
    double latency_ms_per_tok;
};

// Helper: Get current timestamp formatted as YYYY-MM-DD HH:MM:SS
std::string get_current_timestamp() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

// Helper: Check if a file exists
bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

// Helper: Ensure directory exists
void ensure_directory(const std::string& dir) {
    mkdir(dir.c_str(), 0755);
}

// Save results to persistent CSV history file
void save_to_csv(const std::string& filepath, const std::vector<BenchmarkResult>& results) {
    bool is_new_file = !file_exists(filepath);
    std::ofstream out(filepath, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Warning: Could not open " << filepath << " for writing.\n";
        return;
    }

    if (is_new_file) {
        out << "timestamp,model_name,model_size_mb,threads,dim,n_layers,n_heads,"
            << "prompt_tokens,prefill_ms,prefill_tok_s,decode_tokens,decode_ms,"
            << "decode_tok_s,latency_ms_per_tok\n";
    }

    for (const auto& r : results) {
        out << r.timestamp << ","
            << r.model_name << ","
            << std::fixed << std::setprecision(2) << r.model_size_mb << ","
            << r.num_threads << ","
            << r.dim << ","
            << r.n_layers << ","
            << r.n_heads << ","
            << r.prompt_tokens << ","
            << std::fixed << std::setprecision(2) << r.prefill_ms << ","
            << std::fixed << std::setprecision(2) << r.prefill_tok_s << ","
            << r.decode_tokens << ","
            << std::fixed << std::setprecision(2) << r.decode_ms << ","
            << std::fixed << std::setprecision(2) << r.decode_tok_s << ","
            << std::fixed << std::setprecision(2) << r.latency_ms_per_tok << "\n";
    }
}

// Save latest results as JSON
void save_to_json(const std::string& filepath, const std::vector<BenchmarkResult>& results) {
    std::ofstream out(filepath);
    if (!out.is_open()) return;

    out << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "  {\n";
        out << "    \"timestamp\": \"" << r.timestamp << "\",\n";
        out << "    \"model_name\": \"" << r.model_name << "\",\n";
        out << "    \"model_size_mb\": " << r.model_size_mb << ",\n";
        out << "    \"threads\": " << r.num_threads << ",\n";
        out << "    \"dim\": " << r.dim << ",\n";
        out << "    \"n_layers\": " << r.n_layers << ",\n";
        out << "    \"n_heads\": " << r.n_heads << ",\n";
        out << "    \"prompt_tokens\": " << r.prompt_tokens << ",\n";
        out << "    \"prefill_ms\": " << r.prefill_ms << ",\n";
        out << "    \"prefill_tok_s\": " << r.prefill_tok_s << ",\n";
        out << "    \"decode_tokens\": " << r.decode_tokens << ",\n";
        out << "    \"decode_ms\": " << r.decode_ms << ",\n";
        out << "    \"decode_tok_s\": " << r.decode_tok_s << ",\n";
        out << "    \"latency_ms_per_tok\": " << r.latency_ms_per_tok << "\n";
        out << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "]\n";
}

// Benchmark a single model
BenchmarkResult run_model_benchmark(
    const std::string& model_path,
    const std::string& model_name,
    Tokenizer& tokenizer,
    const std::string& prompt_text,
    int target_decode_tokens
) {
    Model model(model_path);
    const Config& config = model.config;

    struct stat sb;
    stat(model_path.c_str(), &sb);
    double size_mb = static_cast<double>(sb.st_size) / (1024.0 * 1024.0);

    KVCache kv_cache(config);
    RunState state(config);

    // Tokenize benchmark prompt
    std::vector<int> prompt_tokens = tokenizer.tokenize(prompt_text);
    if (prompt_tokens.empty() || prompt_tokens[0] != 1) {
        prompt_tokens.insert(prompt_tokens.begin(), 1); // BOS
    }
    int num_prompt_tokens = static_cast<int>(prompt_tokens.size());

    // 1. Warmup pass (3 prompt tokens + 2 decode steps) to avoid cold page faults
    {
        KVCache warmup_cache(config);
        RunState warmup_state(config);
        forward(1, 0, model, warmup_cache, warmup_state);
        float* warmup_logits = forward(prompt_tokens[1], 1, model, warmup_cache, warmup_state);
        int next_tok = sample_argmax(warmup_logits, std::abs(config.vocab_size));
        forward(next_tok, 2, model, warmup_cache, warmup_state);
    }

    // 2. Measure Prefill Phase
    auto prefill_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_prompt_tokens - 1; ++i) {
        forward(prompt_tokens[i], i, model, kv_cache, state);
    }
    float* last_logits = forward(prompt_tokens[num_prompt_tokens - 1], num_prompt_tokens - 1, model, kv_cache, state);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> prefill_dur = prefill_end - prefill_start;
    double prefill_ms = prefill_dur.count();
    double prefill_tok_s = (num_prompt_tokens / (prefill_ms / 1000.0));

    // First generated token
    int current_token = sample_argmax(last_logits, std::abs(config.vocab_size));

    // 3. Measure Autoregressive Decoding Phase (deterministic greedy decoding)
    int pos = num_prompt_tokens;
    int generated_count = 0;

    auto decode_start = std::chrono::high_resolution_clock::now();

    while (pos < config.seq_len && generated_count < target_decode_tokens) {
        float* logits = forward(current_token, pos, model, kv_cache, state);
        current_token = sample_argmax(logits, std::abs(config.vocab_size));
        pos++;
        generated_count++;
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decode_dur = decode_end - decode_start;
    double decode_ms = decode_dur.count();
    double decode_tok_s = (generated_count / (decode_ms / 1000.0));
    double latency_ms_per_tok = decode_ms / generated_count;

    BenchmarkResult res;
    res.timestamp = get_current_timestamp();
    res.model_name = model_name;
    res.model_size_mb = size_mb;
    res.num_threads = omp_get_max_threads();
    res.dim = config.dim;
    res.n_layers = config.n_layers;
    res.n_heads = config.n_heads;
    res.prompt_tokens = num_prompt_tokens;
    res.prefill_ms = prefill_ms;
    res.prefill_tok_s = prefill_tok_s;
    res.decode_tokens = generated_count;
    res.decode_ms = decode_ms;
    res.decode_tok_s = decode_tok_s;
    res.latency_ms_per_tok = latency_ms_per_tok;

    return res;
}

int main() {
    std::cout << "\n=======================================================================\n";
    std::cout << "                  INFERENCE ENGINE BENCHMARK SUITE\n";
    std::cout << "=======================================================================\n";

    std::string tokenizer_path = "models/tokenizer.bin";
    if (!file_exists(tokenizer_path)) {
        std::cerr << "Error: " << tokenizer_path << " not found!\n";
        return 1;
    }

    Tokenizer tokenizer(32000);
    tokenizer.load(tokenizer_path.c_str());

    std::vector<std::pair<std::string, std::string>> models_to_test = {
        {"models/stories15M.bin", "Stories15M"},
        {"models/stories42M.bin", "Stories42M"},
        {"models/stories110M.bin", "Stories110M"}
    };

    std::string benchmark_prompt =
        "Once upon a time, in a lush green valley surrounded by tall blue mountains, "
        "there lived a kind little hedgehog who loved discovering secret trails.";
    int target_decode_tokens = 100;

    std::vector<BenchmarkResult> results;

    std::cout << "Benchmark Settings:\n";
    std::cout << "  • Hardware Threads:  " << omp_get_max_threads() << "\n";
    std::cout << "  • Target Tokens:     " << target_decode_tokens << " decode steps\n";
    std::cout << "  • Temperature:       0.0 (Greedy Argmax for reproducibility)\n\n";

    for (const auto& [path, name] : models_to_test) {
        if (!file_exists(path)) {
            continue;
        }

        std::cout << "Benchmarking " << name << " (" << path << ")...\n";
        BenchmarkResult r = run_model_benchmark(path, name, tokenizer, benchmark_prompt, target_decode_tokens);
        results.push_back(r);
    }

    if (results.empty()) {
        std::cerr << "No model files found to benchmark in models/\n";
        return 1;
    }

    // Print summary table
    std::cout << "\n====================================================================================\n";
    std::cout << std::left
              << std::setw(14) << "Model"
              << std::setw(10) << "Size"
              << std::setw(16) << "Prefill (tok/s)"
              << std::setw(16) << "Decode (tok/s)"
              << std::setw(16) << "Latency/Token"
              << std::setw(10) << "Threads"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::string size_str = std::to_string(static_cast<int>(std::round(r.model_size_mb))) + " MB";
        std::string prefill_str = std::to_string(static_cast<int>(std::round(r.prefill_tok_s))) + " tok/s";
        std::string decode_str = std::to_string(static_cast<int>(std::round(r.decode_tok_s))) + " tok/s";
        std::string lat_str = std::to_string(r.latency_ms_per_tok).substr(0, 5) + " ms";

        std::cout << std::left
                  << std::setw(14) << r.model_name
                  << std::setw(10) << size_str
                  << std::setw(16) << prefill_str
                  << std::setw(16) << decode_str
                  << std::setw(16) << lat_str
                  << std::setw(10) << r.num_threads
                  << "\n";
    }
    std::cout << "====================================================================================\n\n";

    // Persistence
    std::string results_dir = "benchmark_results";
    ensure_directory(results_dir);

    std::string csv_path = results_dir + "/history.csv";
    std::string json_path = results_dir + "/latest.json";

    save_to_csv(csv_path, results);
    save_to_json(json_path, results);

    std::cout << "Benchmark data saved with timestamp [" << results[0].timestamp << "]:\n";
    std::cout << "  • History: " << csv_path << " (appended)\n";
    std::cout << "  • Latest:  " << json_path << " (overwritten)\n\n";

    return 0;
}
