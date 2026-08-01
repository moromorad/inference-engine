#include "model.h"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath> // For std::abs


Model::Model(const std::string& model_path) {

    // 1. Open the binary file
    fd = open(model_path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error: Could not open " << model_path << std::endl;
        exit(1);
    }

    // 2. Get the exact file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error: Could not get file size." << std::endl;
        close(fd);
        exit(1);
    }
    file_size = sb.st_size;

    // 3. mmap the file directly into RAM
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "Error: mmap failed." << std::endl;
        close(fd);
        exit(1);
    }

    // 4. Extract the Config Header
    Config* file_config = (Config*)data;
    config = *file_config; // Copy the struct data into our class member
    
    std::cout << "--- Model Architecture ---" << std::endl;
    std::cout << "dim: " << config.dim << std::endl;
    std::cout << "hidden_dim: " << config.hidden_dim << std::endl;
    std::cout << "n_layers: " << config.n_layers << std::endl;
    std::cout << "n_heads: " << config.n_heads << std::endl;
    std::cout << "n_kv_heads: " << config.n_kv_heads << std::endl;
    
    int vocab_size = std::abs(config.vocab_size); 
    std::cout << "vocab_size: " << vocab_size << std::endl;

    // 5. The Pointer Walk
    int head_size = config.dim / config.n_heads;
    
    // Start our walker exactly after the 28-byte header
    float* ptr = (float*)((char*)data + sizeof(Config));

    weights.token_embedding_table = ptr;
    ptr += vocab_size * config.dim;

    weights.rms_att_weight = ptr;
    ptr += config.n_layers * config.dim;

    weights.wq = ptr;
    ptr += config.n_layers * config.dim * (config.n_heads * head_size);

    weights.wk = ptr;
    ptr += config.n_layers * config.dim * (config.n_kv_heads * head_size);

    weights.wv = ptr;
    ptr += config.n_layers * config.dim * (config.n_kv_heads * head_size);

    weights.wo = ptr;
    ptr += config.n_layers * (config.n_heads * head_size) * config.dim;

    weights.rms_ffn_weight = ptr;
    ptr += config.n_layers * config.dim;

    weights.w1 = ptr;
    ptr += config.n_layers * config.dim * config.hidden_dim;

    weights.w2 = ptr;
    ptr += config.n_layers * config.hidden_dim * config.dim;

    weights.w3 = ptr;
    ptr += config.n_layers * config.dim * config.hidden_dim;

    weights.rms_final_weight = ptr;
    ptr += config.dim;

    unsigned long long bytes_read = (char*)ptr - (char*)data;
    if (bytes_read < file_size) {
        weights.wcls = ptr; 
    } else {
        weights.wcls = weights.token_embedding_table; 
    }

    std::cout << "\nSuccess! mmap complete. All " << file_size / 1024 / 1024 << " MB of weights assigned." << std::endl;
}

Model::~Model() {
    // 6. Clean up
    // This runs automatically when the Model object is destroyed.
    if (data != MAP_FAILED && data != nullptr) {
        munmap(data, file_size);
    }
    if (fd != -1) {
        close(fd);
    }
}