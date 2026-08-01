#include <iostream>
#include "model.h"
#include "tokenizer.h"


int main() {
    std::cout << "Booting Inference Engine...\n\n";

    // 1. Initialize the Model (Triggers the constructor and mmap)
    Model model("models/stories15M.bin");

    // 2. Initialize the Tokenizer
    Tokenizer tokenizer(std::abs(model.config.vocab_size));
    tokenizer.load("models/tokenizer.bin");

    // 3. Forward Pass (Next steps!)

    return 0; // Model destructor automatically unmaps memory here
}