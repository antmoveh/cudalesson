#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__device__ int d_x = 1;
__device__ int d_y[2];

__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel(void){
    printf("Constant data c_data = %.2f.\n", c_data);
}


int main(int argc, char **argv) {

    int devID = 0;
    cudaDeviceProp deviceProps;
    
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, devID));

    std::cout <<"运行GPU设备"<< deviceProps.name << std::endl;

    float h_data = 8.8f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));

    printf("Constant data h_data = %2.f\n", sizeof(float));

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}