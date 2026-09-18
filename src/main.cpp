#include "engine.h"
#include "model.h"
#include "tokenizer.h"
#include <iostream>
#include <string>

int main() {
  std::cout << "========================================\n";
  std::cout << "       Stories110M Inference Engine\n";
  std::cout << "========================================\n\n";

  // 1. Initialize the Model (loads weights via mmap)
  Model model("models/stories110M.bin");

  // 2. Initialize the Tokenizer
  Tokenizer tokenizer(std::abs(model.config.vocab_size));
  tokenizer.load("models/tokenizer.bin");

  std::cout << "\nModel and Tokenizer loaded successfully!\n";
  std::cout << "Enter prompt to generate a story (type 'exit' or press Ctrl+D "
               "to quit):\n";

  GenerationConfig gen_config;
  gen_config.max_new_tokens = 256;
  gen_config.temperature = 0.8f;
  gen_config.top_k = 40;

  while (true) {
    std::cout << "\n> ";
    std::string prompt;
    if (!std::getline(std::cin, prompt) || prompt == "exit" ||
        prompt == "quit") {
      break;
    }

    if (prompt.empty()) {
      continue;
    }

    std::cout << "\n--- Story ---\n";
    generate(prompt, model, tokenizer, gen_config, /*stream=*/true);
    std::cout << "-------------\n";
  }

  std::cout << "\nShutting down engine. Goodbye!\n";
  return 0;
}