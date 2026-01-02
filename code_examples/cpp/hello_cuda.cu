#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vector_add(int *vec_a, int* vec_b, int* vec_c, long int vec_size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < vec_size){
        vec_c[index] = vec_a[index] + vec_b[index];
    }
}

int main(){
    // set input data size
    long int vec_size = 1e8;

    // declare input vectors on host
    std::vector<int> h_data_a(vec_size);
    std::vector<int> h_data_b(vec_size);
    std::vector<int> h_data_c(vec_size);

    // fill input vectors with random data
    for(int i = 0; i< vec_size; i++){
        h_data_a[i] = rand();
        h_data_b[i] = rand();
    }

    // pointers to data on the device
    int* d_data_a;
    int* d_data_b;
    int* d_data_c;

    // allocate mem on device
    cudaMalloc(&d_data_a, vec_size*sizeof(int));
    cudaMalloc(&d_data_b, vec_size*sizeof(int));
    cudaMalloc(&d_data_c, vec_size*sizeof(int));

    // copy data from host to device
    cudaMemcpy(d_data_a, h_data_a.data(), vec_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_b, h_data_b.data(), vec_size*sizeof(int), cudaMemcpyHostToDevice);

    // launch the kernel
    int n_blocks = vec_size / 512;
    vector_add<<<1024, 512>>>(d_data_a, d_data_b, d_data_c, vec_size);
    cudaDeviceSynchronize();

    // copy results back from device to host
    cudaMemcpy(h_data_c.data(), d_data_c, vec_size*sizeof(int), cudaMemcpyDeviceToHost);

    // checking first 10 elements
    for(int i=0; i< 10;i++){
        std::cout << h_data_a[i] << " " << h_data_b[i] << " " << h_data_c[i] << "\n";
        if(h_data_c[i] != h_data_a[i]+h_data_b[i]){
            std::cout << "Mismatch\n";
        }
    }

    // free allocated mem on device
    cudaFree(d_data_a);
    cudaFree(d_data_b);
    cudaFree(d_data_c);
}