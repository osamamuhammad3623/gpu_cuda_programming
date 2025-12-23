#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Tile width is usually 16 or 32 based on hardware limits
#define TILE_WIDTH 16

// CUDA Kernel for Tiled Matrix Multiplication
__global__ void matrixMulTiledGPU(const float* A, const float* B, float* C, int N) {
    // 1. Declare Shared Memory for tiles
    // These are local to each thread block and much faster than global memory
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Row and Column index for the resulting C element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // 2. Loop over tiles (the 'tile_index' index moves across the A-row and down the B-column)
    for (int tile_index = 0; tile_index < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++tile_index) {

        // 3. Collaborative Load: Load one element per thread into shared memory
        // Check bounds to handle cases where N is not a multiple of TILE_WIDTH
        if (row < N && (tile_index * TILE_WIDTH + tx) < N)
            s_A[ty][tx] = A[row * N + (tile_index * TILE_WIDTH + tx)];
        else
            s_A[ty][tx] = 0.0f; // Padding with zeros for safety

        if (col < N && (tile_index * TILE_WIDTH + ty) < N)
            s_B[ty][tx] = B[(tile_index * TILE_WIDTH + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        // 4. Synchronize: Wait for all threads in the block to finish loading the tile
        __syncthreads();

        // 5. Compute: Use the shared memory tile to calculate partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // 6. Synchronize: Wait for all threads to finish computing before loading next tile
        __syncthreads();
    }

    // 7. Write result back to Global Memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 200; 
    const int size = N * N;
    size_t bytes = size * sizeof(float);

    // Initialize Host Data
    std::vector<float> h_A(size), h_B(size), h_C_GPU(size);
    for (int i = 0; i < size; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // 8. Configure Grid and Block dimensions
    // For tiling, we use a 2D block matching the TILE_WIDTH
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulTiledGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    std::cout << "Matrix Size: " << N << "x" << N << "\n";
    std::cout << "Tiled GPU Kernel Time: " << gpuTime << " ms" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}