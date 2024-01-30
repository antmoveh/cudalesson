#include <stdio.h>

int main(void) {
    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError error = cudaGetDeviceCount(&iDeviceCount);

    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found \n");
        exit(-1);
    } else {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    int iDev = 0;

    error = cudaSetDevice(iDev);
    if (error != cudaSuccess) {
        printf("fail to set GPU 0 for computing. \n");
        exit(-1);
    } else {
        printf("set GPU 0 fro computing.\n");
    }
}