#include "packet_capture.h"
#include <iostream>
#include <pcap.h>

// Static tracker for true Inter-Arrival Time (IAT) calculation
static struct timeval last_ts = {0, 0};

void packet_handler(u_char* user_data, const struct pcap_pkthdr* pkthdr, const u_char* packet) {
    LockFreeRingBuffer* ring_buffer = reinterpret_cast<LockFreeRingBuffer*>(user_data);

    if (pkthdr->caplen < 34) return; 

    uint8_t protocol = packet[23];
    if (protocol == 6) { // TCP Protocol
        uint8_t ip_ihl = (packet[14] & 0x0F) * 4; 
        
        if (pkthdr->caplen < 14 + ip_ihl + 14) return; 

        uint8_t tcp_flags = packet[14 + ip_ihl + 13];
        
        // Calculate exact IAT in milliseconds
        float iat = 0.0f;
        if (last_ts.tv_sec != 0) {
            long seconds = pkthdr->ts.tv_sec - last_ts.tv_sec;
            long useconds = pkthdr->ts.tv_usec - last_ts.tv_usec;
            iat = (seconds * 1000.0f) + (useconds / 1000.0f);
            if (iat < 0.0f) iat = 0.0f; // Handle clock skew
        }
        last_ts = pkthdr->ts;
        
        PacketFeature feature;
        feature.iat = iat; 
        feature.tcp_flags = static_cast<float>(tcp_flags);
        feature.entropy = 4.5f; // Placeholder for payload entropy
        feature.padding = 0.0f;

        ring_buffer->push(feature);
    }
}

void start_capture(const char* device, LockFreeRingBuffer* rb) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* handle = pcap_open_live(device, BUFSIZ, 1, 10, errbuf);
    
    if (handle == nullptr) {
        std::cerr << "Could not open device " << device << ": " << errbuf << std::endl;
        return;
    }

    pcap_loop(handle, 0, packet_handler, reinterpret_cast<u_char*>(rb));
    pcap_close(handle);
}