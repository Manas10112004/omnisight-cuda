#pragma once
#include <atomic>
#include <vector>
#include <cstddef>

struct alignas(16) PacketFeature {
    float iat;
    float entropy;
    float tcp_flags;
    float padding; // 16-byte alignment for coalesced GPU memory access
};

class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t capacity) 
        : capacity_(capacity), buffer_(capacity) {}

    bool push(const PacketFeature& feature) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) % capacity_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[tail] = feature;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(PacketFeature& feature) {
        size_t head = head_.load(std::memory_order_relaxed);
        
        if (head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        feature = buffer_[head];
        head_.store((head + 1) % capacity_, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        return (tail >= head) ? (tail - head) : (capacity_ - head + tail);
    }

private:
    size_t capacity_;
    std::vector<PacketFeature> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};