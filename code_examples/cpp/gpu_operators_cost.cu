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

// CUDA Kernel: Each thread adds the scalar value to each element
__global__ void addVectorGPU(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] += scalar;
    }
}

// CUDA Kernel: Each thread divides each element by the scalar value
__global__ void divideVectorGPU(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] /= scalar;
    }
}

// CUDA Kernel: Each thread checks if each element is even
__global__ void moduloVectorGPU(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = ((int)data[i] % 2 == 0);
    }
}

// CUDA Kernel: Each thread checks if each element is even (optimized)
__global__ void isEvenVectorGPU(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = ((int)data[i] & 1 == 0);
    }
}

int main() {
    const int N = 1e7;
    const float scalar = 3014.128f;
    size_t size = N * sizeof(float);

    // 1. Initialize Host Data
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++) h_data[i] = static_cast<float>(rand()) / RAND_MAX;

    // --- GPU EXECUTION ---
    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    // Setup Timing Events
    cudaEvent_t start, stop;
    float gpuTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // scaleVectorGPU kernel
    cudaEventRecord(start);
    scaleVectorGPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Vector scaling Kernel Time: " << gpuTime << " ms" << std::endl;

    // addVectorGPU kernel
    cudaEventRecord(start);
    addVectorGPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Vector addition Kernel Time: " << gpuTime << " ms" << std::endl;

    // divideVectorGPU kernel
    cudaEventRecord(start);
    divideVectorGPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Vector division Kernel Time: " << gpuTime << " ms" << std::endl;

    // moduloVectorGPU kernel
    cudaEventRecord(start);
    moduloVectorGPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Vector modulo Kernel Time: " << gpuTime << " ms" << std::endl;

    // isEvenVectorGPU kernel
    cudaEventRecord(start);
    isEvenVectorGPU<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "Vector isEven Kernel Time: " << gpuTime << " ms" << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}