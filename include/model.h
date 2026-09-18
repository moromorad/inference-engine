#pragma once

#include "config.h"
#include <cstddef>
#include <string>

// ============================================================================
// MODEL LOADER & MEMORY MAPPER
// ============================================================================

/**
 * Model Weight Manager and Memory Mapper.
 *
 * Responsible for memory-mapping (mmap) binary model files from disk directly
 * into the process address space. This provides zero-copy, instantaneous model
 * loading and read-only memory sharing without heap copying overhead.
 *
 * Responsibilities:
 *   - Parses the 28-byte Config header.
 *   - Performs the pointer walk to link TransformerWeights to contiguous weight buffers.
 *   - Handles weight tying (detecting whether wcls shares memory with token_embedding_table).
 *   - Automatically cleans up the mapped memory region and file descriptor upon destruction.
 */
class Model {
public:
    // Parsed architecture configuration
    Config config;

    // Direct pointers to all weight tensors in the mapped memory
    TransformerWeights weights;

    /**
     * Constructs the Model and memory-maps the binary weights file.
     *
     * @param model_path Path to the binary model file (e.g., "models/stories15M.bin").
     * @throws Exits with error code if the file cannot be opened or mapped.
     */
    explicit Model(const std::string& model_path);

    /**
     * Destructor: Automatically unmaps virtual memory (munmap) and closes the open file descriptor.
     */
    ~Model();

    // Prevent accidental copying to avoid double-unmapping of memory
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Allow moving resources
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

private:
    // Low-level POSIX file descriptor
    int fd;

    // Raw pointer to the start of the mmap'd memory block
    void* data;

    // Total file size in bytes
    size_t file_size;
};