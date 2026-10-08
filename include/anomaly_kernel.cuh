#pragma once
#include <cuda_runtime.h>

// Launches the Z-score normalization kernel
void launch_feature_normalization(float* d_X, int batch_size, int input_dim, cudaStream_t stream);

// Launches the Tiled Shared-Memory Autoencoder (Encoder + Decoder + MSE)
void launch_autoencoder_inference(
    const float* d_X, 
    const float* d_W_enc, 
    const float* d_W_dec,
    float* d_Hidden,
    float* d_Out_Scores, 
    int batch_size, 
    int input_dim, 
    int hidden_dim, 
    cudaStream_t stream
);