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

    // Prompt the user
    std::cout << "Enter string for tokenization ";
    std::string text;
    std::getline(std::cin, text);

    // Tokenize the string
    std::vector<int> tokens = tokenizer.tokenize(text);

    // Print the tokens
    std::cout << "\nTokens:\n";
    for (int token : tokens) {
        std::cout << "  [" << token << "] \"" << tokenizer.vocab[token] << "\"\n";
    }
    std::cout << std::endl;

    // 3. Forward Pass (Next steps!)

    return 0; // Model destructor automatically unmaps memory here
}