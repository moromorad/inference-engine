#ifndef CONFIG_H
#define CONFIG_H

// Matches the 28-byte header in the AI's binary file
struct Config {
    int dim;        // Token embedding dimension
    int hidden_dim; // Feed-Forward network internal size
    int n_layers;   // Number of Transformer blocks
    int n_heads;    // Number of query attention heads
    int n_kv_heads; // Number of key/value attention heads
    int vocab_size; // Size of the dictionary
    int seq_len;    // Maximum context length for the KV cache
};

// Pointers that will link directly to the AI's math matrices
struct TransformerWeights {
    float* token_embedding_table; 
    
    // Attention weights
    float* rms_att_weight; 
    float* wq;             
    float* wk;             
    float* wv;             
    float* wo;             
    
    // Feed-Forward weights
    float* rms_ffn_weight; 
    float* w1;             
    float* w2;             
    float* w3;             
    
    // Final normalization and output classifier
    float* rms_final_weight; 
    float* wcls;             
};

#endif // CONFIG_H