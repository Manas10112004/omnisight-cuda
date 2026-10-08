#include "anomaly_kernel.cuh"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void autoencoder_inference_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ Y,
    int batch_size, 
    int input_dim, 
    int hidden_dim) 
{
    __shared__ float s_X[TILE_SIZE][TILE_SIZE];
    __shared__ float s_W[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float accumulated_val = 0.0f;

    for (int p = 0; p < (input_dim + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        if (row < batch_size && p * TILE_SIZE + threadIdx.x < input_dim) {
            s_X[threadIdx.y][threadIdx.x] = X[row * input_dim + p * TILE_SIZE + threadIdx.x];
        } else {
            s_X[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (p * TILE_SIZE + threadIdx.y < input_dim && col < hidden_dim) {
            s_W[threadIdx.y][threadIdx.x] = W[(p * TILE_SIZE + threadIdx.y) * hidden_dim + col];
        } else {
            s_W[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            accumulated_val += s_X[threadIdx.y][i] * s_W[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < batch_size && col < hidden_dim) {
        Y[row * hidden_dim + col] = fmaxf(0.0f, accumulated_val);
    }
}

__global__ void decoder_mse_kernel(
    const float* __restrict__ Original_X,
    const float* __restrict__ Hidden_Z, 
    const float* __restrict__ W_dec, 
    float* __restrict__ Anomaly_Scores, 
    int batch_size, 
    int hidden_dim, 
    int input_dim) 
{
    __shared__ float s_Z[TILE_SIZE][TILE_SIZE];
    __shared__ float s_W[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float reconstructed_val = 0.0f;

    for (int p = 0; p < (hidden_dim + TILE_SIZE - 1) / TILE_SIZE; ++p) {
        if (row < batch_size && p * TILE_SIZE + threadIdx.x < hidden_dim) {
            s_Z[threadIdx.y][threadIdx.x] = Hidden_Z[row * hidden_dim + p * TILE_SIZE + threadIdx.x];
        } else {
            s_Z[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (p * TILE_SIZE + threadIdx.y < hidden_dim && col < input_dim) {
            s_W[threadIdx.y][threadIdx.x] = W_dec[(p * TILE_SIZE + threadIdx.y) * input_dim + col];
        } else {
            s_W[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            reconstructed_val += s_Z[threadIdx.y][i] * s_W[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < batch_size && col < input_dim) {
        float diff = Original_X[row * input_dim + col] - reconstructed_val;
        float squared_error = diff * diff;
        
        atomicAdd(&Anomaly_Scores[row], squared_error / input_dim);
    }
}

void launch_autoencoder_inference(
    const float* d_X, const float* d_W_enc, const float* d_W_dec,
    float* d_Hidden, float* d_Out_Scores, int batch_size, int input_dim, int hidden_dim, cudaStream_t stream) 
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid_enc((hidden_dim + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    dim3 grid_dec((input_dim + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    
    cudaMemsetAsync(d_Out_Scores, 0, batch_size * sizeof(float), stream);

    autoencoder_inference_kernel<<<grid_enc, block, 0, stream>>>(d_X, d_W_enc, d_Hidden, batch_size, input_dim, hidden_dim);
    decoder_mse_kernel<<<grid_dec, block, 0, stream>>>(d_X, d_Hidden, d_W_dec, d_Out_Scores, batch_size, hidden_dim, input_dim);
}