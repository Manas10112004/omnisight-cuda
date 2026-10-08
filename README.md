# OmniSight CUDA

A real-time, GPU-accelerated network intrusion detection engine built from scratch using C++, CUDA, and PyTorch. 

OmniSight intercepts live network traffic via Npcap, extracts temporal and protocol features, and evaluates them against a neural network autoencoder. By leveraging lock-free data structures and dual-stream CUDA double-buffering, the engine achieves real-time anomaly detection with virtually zero packet-drop overhead.

## Architecture & Features

*   **Lock-Free Ring Buffer:** Implemented a thread-safe, lock-free C++ ring buffer (`std::atomic`) to decouple the Npcap packet capture thread from the GPU processing thread, eliminating mutex bottlenecks during high-throughput network floods.
*   **Dual-Stream Double-Buffering:** Utilizes pinned host memory (`cudaHostAlloc`) and concurrent CUDA streams to overlap CPU-to-GPU data transfers with active kernel execution. While Stream 0 computes inference on batch *N*, Stream 1 transfers batch *N+1*.
*   **Custom CUDA Inference Engine:** Bypasses heavy ML framework overhead by implementing the Autoencoder inference and Mean Squared Error (MSE) loss calculation directly in custom CUDA C++ kernels.
*   **PyTorch Baseline Training:** Includes an offline Python pipeline that parses baseline `.pcap` files, trains a 64-dimensional Autoencoder to map "normal" traffic patterns, and exports raw binary weights (`weights.bin`) for native ingestion by the C++ engine.

## Technology Stack

*   **Systems:** C++17, CMake, Ninja, Visual Studio Build Tools
*   **Parallel Computing:** CUDA C++, NVCC
*   **Networking:** Npcap (libpcap)
*   **Machine Learning:** Python, PyTorch, Scapy

## Project Structure

```text
omnisight-cuda/
├── CMakeLists.txt
├── include/                 # C++ and CUDA headers
│   ├── anomaly_kernel.cuh
│   ├── cuda_pipeline.cuh
│   ├── packet_capture.h
│   └── ring_buffer.h
├── src/
│   ├── main.cpp             # Engine entry point and loop
│   ├── network/             # Npcap capture logic
│   └── pipeline/            # Weight loading and stream sync
├── cuda/                    # GPU Kernels
│   ├── anomaly_kernel.cu    # Matrix multiplication and MSE computation
│   └── memory_stream.cu     # Pinned memory allocation and async transfers
└── training/
    └── train_autoencoder.py # PyTorch baseline training script