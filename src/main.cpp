#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "config.h"

int main() {
    const char* model_path = "models/stories15M.bin";

    // 1. Open the binary file
    int fd = open(model_path, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error: Could not open " << model_path << std::endl;
        return 1;
    }

    // 2. Get the exact file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error: Could not get file size." << std::endl;
        close(fd);
        return 1;
    }

    // 3. mmap the file directly into RAM
    void* data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        std::cerr << "Error: mmap failed." << std::endl;
        close(fd);
        return 1;
    }

    // 4. Extract the Config Header
    // Point a Config struct directly at the first 28 bytes (Byte 0)
    Config* config = (Config*)data;
    
    std::cout << "--- Model Architecture ---" << std::endl;
    std::cout << "dim: " << config->dim << std::endl;
    std::cout << "hidden_dim: " << config->hidden_dim << std::endl;
    std::cout << "n_layers: " << config->n_layers << std::endl;
    std::cout << "n_heads: " << config->n_heads << std::endl;
    std::cout << "n_kv_heads: " << config->n_kv_heads << std::endl;
    
    // Wait, some models store vocab size as negative to indicate weight sharing. 
    // We use absolute value to be safe.
    int vocab_size = abs(config->vocab_size); 
    std::cout << "vocab_size: " << vocab_size << std::endl;

    // 5. The Pointer Walk
    TransformerWeights weights;
    
    // The size of a single attention head is the total dimension divided by the number of heads
    int head_size = config->dim / config->n_heads;
    
    // Start our walker exactly after the 28-byte header
    float* ptr = (float*)((char*)data + sizeof(Config));

    // Assign the pointer, then advance the walker by the exact matrix size
    weights.token_embedding_table = ptr;
    ptr += vocab_size * config->dim;

    weights.rms_att_weight = ptr;
    ptr += config->n_layers * config->dim;

    weights.wq = ptr;
    ptr += config->n_layers * config->dim * (config->n_heads * head_size);

    weights.wk = ptr;
    ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);

    weights.wv = ptr;
    ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);

    weights.wo = ptr;
    ptr += config->n_layers * (config->n_heads * head_size) * config->dim;

    weights.rms_ffn_weight = ptr;
    ptr += config->n_layers * config->dim;

    weights.w1 = ptr;
    ptr += config->n_layers * config->dim * config->hidden_dim;

    weights.w2 = ptr;
    ptr += config->n_layers * config->hidden_dim * config->dim;

    weights.w3 = ptr;
    ptr += config->n_layers * config->dim * config->hidden_dim;

    weights.rms_final_weight = ptr;
    ptr += config->dim;

    // Neat AI trick: Weight Tying.
    // Often, the final classifier matrix (wcls) is identical to the token embedding matrix.
    // To save space, the file just ends here. We check if we've run out of file bytes.
    unsigned long long bytes_read = (char*)ptr - (char*)data;
    if (bytes_read < sb.st_size) {
        weights.wcls = ptr; // It has its own classifier weights
    } else {
        weights.wcls = weights.token_embedding_table; // Share the embedding weights
    }

    std::cout << "\nSuccess! mmap complete. All " << sb.st_size / 1024 / 1024 << " MB of weights assigned." << std::endl;

    // 6. Clean up (For later: we will actually keep this open when running the AI)
    munmap(data, sb.st_size);
    close(fd);

    return 0;
}