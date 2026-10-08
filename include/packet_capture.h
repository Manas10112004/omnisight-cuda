#pragma once
#include "ring_buffer.h"

// Starts the libpcap loop on the dynamically selected interface
void start_capture(const char* device, LockFreeRingBuffer* rb);