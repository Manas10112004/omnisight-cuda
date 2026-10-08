#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

std::vector<float> load_binary_weights(const std::string& filepath, size_t expected_elements);

class CudaPipeline {
public:
    CudaPipeline(int batch_size, int input_dim, int hidden_dim);
    ~CudaPipeline();

    void submit_batch(float* batch_data, bool use_stream1);
    void sync_stream(bool use_stream1);
    void load_weights(const float* host_weights, size_t size);

    float* get_h_pinned_X0() const { return h_pinned_X0_; }
    float* get_h_pinned_X1() const { return h_pinned_X1_; }
    float* get_h_pinned_Y0() const { return h_pinned_Y0_; }
    float* get_h_pinned_Y1() const { return h_pinned_Y1_; }

private:
    int batch_size_;
    int input_dim_;
    int hidden_dim_;

    // Pinned Host Memory for inputs and outputs (Double-Buffered)
    float *h_pinned_X0_, *h_pinned_X1_;
    float *h_pinned_Y0_, *h_pinned_Y1_;

    // Device Memory per stream to avoid data races during concurrent execution
    float *d_X0_, *d_X1_;
    float *d_Hidden0_, *d_Hidden1_;
    float *d_Y0_, *d_Y1_;

    // Model weights in device memory
    float *d_W_enc_, *d_W_dec_;

    // CUDA Streams
    cudaStream_t stream0_, stream1_;
};