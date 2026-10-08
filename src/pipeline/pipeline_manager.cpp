#include "cuda_pipeline.cuh"
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

std::vector<float> load_binary_weights(const std::string& filepath, size_t expected_elements) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open weights file: " + filepath);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size != static_cast<std::streamsize>(expected_elements * sizeof(float))) {
        throw std::runtime_error("Weight file size mismatch. Expected " + 
                                 std::to_string(expected_elements * sizeof(float)) + 
                                 " bytes, got " + std::to_string(size));
    }

    std::vector<float> weights(expected_elements);
    if (!file.read(reinterpret_cast<char*>(weights.data()), size)) {
        throw std::runtime_error("Error reading binary weights.");
    }

    return weights;
}

void CudaPipeline::load_weights(const float* host_weights, size_t size) {
    size_t half_size = size / 2;
    cudaMemcpy(d_W_enc_, host_weights, half_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_dec_, host_weights + half_size, half_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}