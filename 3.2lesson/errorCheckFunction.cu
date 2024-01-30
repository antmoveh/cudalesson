#include <stdio.h>
#include "../tools/common.cuh"




int main(void) {
    float *fpHost_A;
    fpHost_A = (float *)malloc(4);
    memset(fpHost_A, 0, 4);

    float *fpDevice_A;
    cudaError_t error = ErrorCheck(cudaMalloc((float**)&fpDevice_A, 4), __FILE__, __LINE__);
    cudaMemset(fpDevice_A, 0, 4);
    // 这里是有错误的
    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // 捕获核函数错误
    // addFromGPU<<<grid, block>>>(f1, f2, iElemCount);
    // ErrorCheck(cudaGetLastError(), __FILE__, __LINE__)

    free(fpHost_A);
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}