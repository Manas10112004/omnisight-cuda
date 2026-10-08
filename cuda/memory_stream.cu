#include "cuda_pipeline.cuh"
#include "anomaly_kernel.cuh"
#include <iostream>

CudaPipeline::CudaPipeline(int batch_size, int input_dim, int hidden_dim) 
    : batch_size_(batch_size), input_dim_(input_dim), hidden_dim_(hidden_dim)
{
    size_t in_bytes = batch_size_ * input_dim_ * sizeof(float);
    size_t out_bytes = batch_size_ * sizeof(float);
    size_t hidden_bytes = batch_size_ * hidden_dim_ * sizeof(float);

    // Pinned host buffers for both streams
    cudaHostAlloc((void**)&h_pinned_X0_, in_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_pinned_X1_, in_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_pinned_Y0_, out_bytes, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_pinned_Y1_, out_bytes, cudaHostAllocDefault);

    // Independent device buffers for both streams
    cudaMalloc((void**)&d_X0_, in_bytes);
    cudaMalloc((void**)&d_X1_, in_bytes);
    cudaMalloc((void**)&d_Hidden0_, hidden_bytes);
    cudaMalloc((void**)&d_Hidden1_, hidden_bytes);
    cudaMalloc((void**)&d_Y0_, out_bytes);
    cudaMalloc((void**)&d_Y1_, out_bytes);

    // Weights
    cudaMalloc((void**)&d_W_enc_, input_dim_ * hidden_dim_ * sizeof(float)); 
    cudaMalloc((void**)&d_W_dec_, hidden_dim_ * input_dim_ * sizeof(float)); 

    cudaStreamCreate(&stream0_);
    cudaStreamCreate(&stream1_);
}

CudaPipeline::~CudaPipeline() {
    cudaFreeHost(h_pinned_X0_);
    cudaFreeHost(h_pinned_X1_);
    cudaFreeHost(h_pinned_Y0_);
    cudaFreeHost(h_pinned_Y1_);
    cudaFree(d_X0_);
    cudaFree(d_X1_);
    cudaFree(d_Hidden0_);
    cudaFree(d_Hidden1_);
    cudaFree(d_Y0_);
    cudaFree(d_Y1_);
    cudaFree(d_W_enc_);
    cudaFree(d_W_dec_);
    cudaStreamDestroy(stream0_);
    cudaStreamDestroy(stream1_);
}

void CudaPipeline::sync_stream(bool use_stream1) {
    cudaStreamSynchronize(use_stream1 ? stream1_ : stream0_);
}

void CudaPipeline::submit_batch(float* batch_data, bool use_stream1) {
    size_t in_bytes = batch_size_ * input_dim_ * sizeof(float);
    size_t out_bytes = batch_size_ * sizeof(float);

    cudaStream_t current_stream = use_stream1 ? stream1_ : stream0_;
    float* current_dX = use_stream1 ? d_X1_ : d_X0_;
    float* current_dHidden = use_stream1 ? d_Hidden1_ : d_Hidden0_;
    float* current_dY = use_stream1 ? d_Y1_ : d_Y0_;
    float* current_hY = use_stream1 ? h_pinned_Y1_ : h_pinned_Y0_;

    // 1. Asynchronous Host-to-Device transfer
    cudaMemcpyAsync(current_dX, batch_data, in_bytes, cudaMemcpyHostToDevice, current_stream);

    // 2. Feature normalization kernel
    launch_feature_normalization(current_dX, batch_size_, input_dim_, current_stream);

    // 3. Autoencoder inference & MSE reconstruction kernel
    launch_autoencoder_inference(
        current_dX, d_W_enc_, d_W_dec_, current_dHidden, current_dY, 
        batch_size_, input_dim_, hidden_dim_, current_stream
    );

    // 4. Asynchronous Device-to-Host transfer of anomaly scores
    cudaMemcpyAsync(current_hY, current_dY, out_bytes, cudaMemcpyDeviceToHost, current_stream);
}