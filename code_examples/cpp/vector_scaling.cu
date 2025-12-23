#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA Kernel: Each thread scales one element
__global__ void scaleVectorGPU(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= scalar;
    }
}

// CPU Function
void scaleVectorCPU(std::vector<float>& data, float scalar) {
    for (float& val : data) {
        val *= scalar;
    }
}

int main() {
    const int N = 1e10;
    const float scalar = 3014.128f;
    size_t size = N * sizeof(float);

    // 1. Initialize Host Data
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++) h_data[i] = static_cast<float>(rand()) / RAND_MAX;

    // --- CPU EXECUTION ---
    auto startCPU = std::chrono::high_resolution_clock::now();
    scaleVectorCPU(h_data, scalar);
    auto endCPU = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> cpuTime = endCPU - startCPU;
    std::cout << "CPU Time: " << cpuTime.count() << " ms" << std::endl;

    // --- GPU EXECUTION ---
    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    // Setup Timing Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    scaleVectorGPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "GPU Kernel Time: " << gpuTime << " ms" << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}