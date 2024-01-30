#pragma once
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call) __kernelCheck(call, __FILE__, __LINE__)


static void __cudaCheck(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE: %s, DETAIL: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}


static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}



cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber){
    if (error_code != cudaSuccess) {
        printf("CUDA error: code=%d, name=%s, description=%s file=%s, line%d\r\n",
        error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}


void setGPU() {
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);

    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found \n");
        exit(-1);
    } else {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    int iDev = 0;

    error = ErrorCheck(cudaSetDevice(iDev), __FILE__, __LINE__);
    if (error != cudaSuccess) {
        printf("fail to set GPU 0 for computing. \n");
        exit(-1);
    } else {
        printf("set GPU 0 fro computing.\n");
    }
}

