import numpy as np
from numba import cuda
import math
import time

# 1. Define the CUDA Kernel in Pure Python
@cuda.jit
def matrix_mul_gpu(A, B, C, N):
    # Calculate the absolute row and column index for this thread
    row, col = cuda.grid(2)

    # Boundary check (same as your if row < N && col < N)
    if row < N and col < N:
        tmp = 0.0
        for k in range(N):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def main():
    # Matrix dimension
    N = 400
    
    # Initialize Host Data
    # Numba works seamlessly with NumPy arrays
    h_A = np.random.random((N, N)).astype(np.float32)
    h_B = np.random.random((N, N)).astype(np.float32)
    h_C = np.zeros((N, N), dtype=np.float32)

    # --- CPU EXECUTION (for verification) ---
    start_cpu = time.time()
    h_C_cpu = np.dot(h_A, h_B)
    print(f"Matrix Size: {N}x{N}")
    print(f"CPU Time (NumPy Dot): {(time.time() - start_cpu) * 1000:.2f} ms")

    # --- GPU EXECUTION ---
    # Transfer data to the Device (GPU)
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.device_array((N, N), dtype=np.float32)

    # Configure the blocks and grids
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(N / threads_per_block[0])
    blocks_per_grid_y = math.ceil(N / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Warm-up (Numba compiles the kernel on the first call)
    matrix_mul_gpu[blocks_per_grid, threads_per_block](d_A, d_B, d_C, N)
    cuda.synchronize()

    # Timing the GPU kernel
    start_gpu = time.time()
    matrix_mul_gpu[blocks_per_grid, threads_per_block](d_A, d_B, d_C, N)
    cuda.synchronize()
    print(f"GPU Kernel Time: {(time.time() - start_gpu) * 1000:.2f} ms")

    # Copy result back to Host
    h_C_gpu = d_C.copy_to_host()

    # Verification
    if np.allclose(h_C_cpu, h_C_gpu, atol=1e-5):
        print("Success: GPU result matches CPU result!")
    else:
        print("Error: Results do not match.")

if __name__ == "__main__":
    main()