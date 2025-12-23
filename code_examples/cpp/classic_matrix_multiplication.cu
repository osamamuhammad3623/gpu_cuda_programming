#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulGPU(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU Function for Matrix Multiplication
void matrixMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 400; // NxN matrices
    const int size = N * N;
    size_t bytes = size * sizeof(float);

    // 1. Initialize Host Data
    std::vector<float> h_A(size), h_B(size), h_C_CPU(size), h_C_GPU(size);
    for (int i = 0; i < size; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // --- CPU EXECUTION ---
    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_CPU, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> cpuTime = endCPU - startCPU;
    std::cout << "Matrix Size: " << N << "x" << N << "\n";
    std::cout << "CPU Time: " << cpuTime.count() << " ms" << std::endl;

    // --- GPU EXECUTION ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Setup Timing Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulGPU<<<20, 512>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    std::cout << "GPU Kernel Time: " << gpuTime << " ms" << std::endl;

    // Copy result back to verify (optional)
    cudaMemcpy(h_C_GPU.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}