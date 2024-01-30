#include <stdio.h>
#include "../tools/common.cuh"

int main(void) {
    int device_id = 0;

    cudaDeviceProp deviceProp;
    ErrorCheck(cudaGetDeviceProperties(&deviceProp, device_id), __FILE__, __LINE__);
    printf("Device id: %d\n", device_id);
    printf("Device Name: %s\n", deviceProp.name);
    printf("Total Global Memory: %lu\n", deviceProp.totalGlobalMem);
    printf("Shared Memory per Block: %lu\n", deviceProp.sharedMemPerBlock);
    printf("Warp Size: %d\n", deviceProp.warpSize);
    printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Threads Dim: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max Grid Size: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Clock Rate: %d\n", deviceProp.clockRate);
    printf("Total Constant Memory: %lu\n", deviceProp.totalConstMem);
    printf("Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
    printf("Kernel Execution Timeout Enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("Integrated: %s\n", deviceProp.integrated ? "Yes" : "No");
    printf("Can Map Host Memory: %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("Compute Mode: %d\n", deviceProp.computeMode);
    printf("Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
    printf("ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
    printf("PCI Bus ID: %d\n", deviceProp.pciBusID);
    printf("PCI Device ID: %d\n", deviceProp.pciDeviceID);
    printf("TCC Driver: %s\n", deviceProp.tccDriver ? "Yes" : "No");

    return 0;
}