#include "ring_buffer.h"
#include "packet_capture.h"
#include "cuda_pipeline.cuh"
#include <thread>
#include <iostream>
#include <fstream>
#include <pcap.h>
#include <chrono>
#include <iomanip>

#define BATCH_SIZE 32  
#define INPUT_DIM 4
#define HIDDEN_DIM 64
#define ANOMALY_THRESHOLD 1.5f

int main() {
    LockFreeRingBuffer ring_buffer(65536);
    CudaPipeline cuda_pipeline(BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);

    const size_t total_weights = (INPUT_DIM * HIDDEN_DIM) + (HIDDEN_DIM * INPUT_DIM);
    try {
        std::vector<float> weights = load_binary_weights("../data/weights.bin", total_weights);
        cuda_pipeline.load_weights(weights.data(), weights.size());
        std::cout << "[+] Loaded model weights from data/weights.bin\n";
    } catch (const std::exception& e) {
        std::cerr << "[!] Initializing dummy weights.\n";
        std::vector<float> dummy_weights(total_weights, 0.05f);
        cuda_pipeline.load_weights(dummy_weights.data(), dummy_weights.size());
    }

    std::ofstream log_file("anomalies.csv", std::ios::app);
    if (log_file.tellp() == 0) {
        log_file << "Timestamp_ms,Stream,Packet_Index,MSE_Score,IAT,Entropy,TCP_Flags,Classification\n";
    }

    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_if_t *alldevs;

    if (pcap_findalldevs(&alldevs, errbuf) == -1 || alldevs == nullptr) {
        std::cerr << "Failed to find devices.\n";
        return 1;
    }

    std::cout << "=========================================\n";
    std::cout << "      OmniSight CUDA - Live Engine       \n";
    std::cout << "=========================================\n";
    
    int i = 0;
    for (pcap_if_t* d = alldevs; d != nullptr; d = d->next) {
        std::cout << ++i << ". " << (d->description ? d->description : d->name) << "\n";
    }

    int inum;
    std::cout << "\nEnter interface number to monitor: ";
    std::cin >> inum;

    pcap_if_t* selected = alldevs;
    for (int j = 1; j < inum; j++) selected = selected->next;
    std::string device_name = selected->name;
    
    std::cout << "\n[+] Binding to " << (selected->description ? selected->description : device_name) << "...\n";
    pcap_freealldevs(alldevs);

    std::thread capture_thread(start_capture, device_name.c_str(), &ring_buffer);

    bool stream_toggle = false;
    bool has_dispatched_stream[2] = { false, false };
    int previous_batch_counts[2] = { 0, 0 };

    PacketFeature current_feature;
    int batch_idx = 0;
    auto last_flush = std::chrono::steady_clock::now();

    std::cout << "[+] Capture active. Awaiting traffic...\n\n";

    while (true) {
        bool has_packet = ring_buffer.pop(current_feature);
        float* target_buffer = stream_toggle ? cuda_pipeline.get_h_pinned_X1() 
                                             : cuda_pipeline.get_h_pinned_X0();

        if (has_packet) {
            target_buffer[batch_idx * INPUT_DIM + 0] = current_feature.iat;
            target_buffer[batch_idx * INPUT_DIM + 1] = current_feature.entropy;
            target_buffer[batch_idx * INPUT_DIM + 2] = current_feature.tcp_flags;
            target_buffer[batch_idx * INPUT_DIM + 3] = current_feature.padding;
            batch_idx++;
        }

        auto now = std::chrono::steady_clock::now();
        bool timeout = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush).count() > 300;

        if (batch_idx == BATCH_SIZE || (batch_idx > 0 && timeout)) {
            for (int pad = batch_idx; pad < BATCH_SIZE; ++pad) {
                target_buffer[pad * INPUT_DIM + 0] = 0.0f;
                target_buffer[pad * INPUT_DIM + 1] = 0.0f;
                target_buffer[pad * INPUT_DIM + 2] = 0.0f;
                target_buffer[pad * INPUT_DIM + 3] = 0.0f;
            }

            int current_stream_id = stream_toggle ? 1 : 0;

            if (has_dispatched_stream[current_stream_id]) {
                cuda_pipeline.sync_stream(stream_toggle);

                float* scores = stream_toggle ? cuda_pipeline.get_h_pinned_Y1() 
                                              : cuda_pipeline.get_h_pinned_Y0();
                float* inputs = stream_toggle ? cuda_pipeline.get_h_pinned_X1() 
                                              : cuda_pipeline.get_h_pinned_X0();

                int active_count = previous_batch_counts[current_stream_id];
                
                // Track max anomaly to prevent console flooding
                float max_score = 0.0f;
                int max_idx = -1;

                auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();

                for (int k = 0; k < active_count; ++k) {
                    float score = scores[k];
                    if (score > max_score) {
                        max_score = score;
                        max_idx = k;
                    }
                    log_file << timestamp_ms << "," << current_stream_id << "," << k << ","
                             << score << "," << inputs[k * INPUT_DIM + 0] << ","
                             << inputs[k * INPUT_DIM + 1] << "," << inputs[k * INPUT_DIM + 2] << ","
                             << ((score > ANOMALY_THRESHOLD) ? "ANOMALY" : "BENIGN") << "\n";
                }
                
                if (max_score > ANOMALY_THRESHOLD && max_idx != -1) {
                    std::cout << "\033[1;31m[ALERT] Peak Anomaly!\033[0m Score: " 
                              << std::fixed << std::setprecision(2) << max_score 
                              << " | IAT: " << inputs[max_idx * INPUT_DIM + 0] << "ms"
                              << " | Flags: 0x" << std::hex << static_cast<int>(inputs[max_idx * INPUT_DIM + 2])
                              << std::dec << "\n";
                }
            }

            cuda_pipeline.submit_batch(target_buffer, stream_toggle);
            has_dispatched_stream[current_stream_id] = true;
            previous_batch_counts[current_stream_id] = batch_idx;

            std::cout << "[GPU] Stream " << current_stream_id 
                      << " <- Dispatched batch (" << std::setw(2) << batch_idx << " packets)\n";

            stream_toggle = !stream_toggle;
            batch_idx = 0;
            last_flush = std::chrono::steady_clock::now();
        }
    }

    capture_thread.join();
    return 0;
}
