#pragma once
#include "config.h" // Assumes Config and TransformerWeights are defined here
#include <string>
#include <cstddef>  // For size_t

class Model {
public:
    Config config;
    TransformerWeights weights;

    // Constructor: Maps the file into memory and assigns pointers
    Model(const std::string& model_path);

    // Destructor: Automatically unmaps memory and closes the file descriptor
    ~Model();

private:
    // Internal variables needed for cleanup
    int fd;
    void* data;
    size_t file_size;
};