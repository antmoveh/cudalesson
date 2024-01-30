#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__global__ void kernel(void){}


int main(int argc, char **argv) {

    int devID = 0;
    cudaDeviceProp cudaDeviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&cudaDeviceProps, devID));
    std::cout << "运行GPU设备:" << cudaDeviceProps.name << std::endl;


    if (cudaDeviceProps.globalL1CacheSupported) {
        std::cout << "支持全局内存L1缓存" << std::endl;
    } else {
        std::cout << "不支持全局内存L1缓存" << std::endl;
    }

    std::cout << "L2缓存大小: " << cudaDeviceProps.l2CacheSize / (1024 * 1024) << "M" << std::endl;

    dim3 block(1);
    dim3 grid(1);

    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

 
    CUDA_CHECK(cudaDeviceReset());
}