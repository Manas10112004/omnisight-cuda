cmake_minimum_required(VERSION 3.20)
project(OmniSightCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for NVIDIA Ampere/Ada/Hopper architectures
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -Xptxas -O3,-v")

find_package(CUDAToolkit REQUIRED)

# Locate Npcap SDK on Windows
if(WIN32)
    set(NPCAP_SDK_DIR "C:/npcap-sdk" CACHE PATH "Path to Npcap SDK")
    
    find_path(PCAP_INCLUDE_DIR NAMES pcap.h PATHS "${NPCAP_SDK_DIR}/Include")
    find_library(PCAP_LIBRARY NAMES wpcap PATHS "${NPCAP_SDK_DIR}/Lib/x64" "${NPCAP_SDK_DIR}/Lib")

    if(NOT PCAP_INCLUDE_DIR OR NOT PCAP_LIBRARY)
        message(FATAL_ERROR "Npcap SDK not found! Ensure it is extracted to C:/npcap-sdk or pass -DNPCAP_SDK_DIR=...")
    endif()

    message(STATUS "Found Npcap Include: ${PCAP_INCLUDE_DIR}")
    message(STATUS "Found Npcap Library: ${PCAP_LIBRARY}")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PCAP REQUIRED libpcap)
    set(PCAP_INCLUDE_DIR ${PCAP_INCLUDE_DIRS})
    set(PCAP_LIBRARY ${PCAP_LIBRARIES})
endif()

include_directories(
    include
    ${PCAP_INCLUDE_DIR}
)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE CUDA_FILES cuda/*.cu)

add_executable(OmniSightCUDA ${SRC_FILES} ${CUDA_FILES})

target_compile_definitions(OmniSightCUDA PRIVATE HAVE_REMOTE WPCAP)

# Link Npcap and CUDA Runtime
target_link_libraries(OmniSightCUDA PRIVATE ${PCAP_LIBRARY} CUDA::cudart)