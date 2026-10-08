#include "anomaly_kernel.cuh"

__constant__ float c_mean[4] = { 10.5f, 3.1f, 0.5f, 0.0f };
__constant__ float c_std[4]  = { 2.1f, 0.8f, 1.2f, 1.0f };

__global__ void normalize_features_kernel(float* __restrict__ X, int batch_size, int input_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size) {
        #pragma unroll
        for (int col = 0; col < input_dim; ++col) {
            int idx = row * input_dim + col;
            X[idx] = (X[idx] - c_mean[col]) / c_std[col];
        }
    }
}

void launch_feature_normalization(float* d_X, int batch_size, int input_dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    normalize_features_kernel<<<blocks, threads, 0, stream>>>(d_X, batch_size, input_dim);
}