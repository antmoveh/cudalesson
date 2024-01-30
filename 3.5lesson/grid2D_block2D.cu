#include <stdio.h>
#include "../tools/common.cuh"

__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {

    // 设置运行GPU
    setGPU();

    int nx = 16;
    int ny = 8;
    int nxy = nx * ny;

    size_t stBytesCount = nxy * sizeof(int);


    int *ipHost_A, *ipHost_B, *ipHost_C;

    ipHost_A = (int *)malloc(stBytesCount);
    ipHost_B = (int *)malloc(stBytesCount);
    ipHost_C = (int *)malloc(stBytesCount);

    if (ipHost_A != NULL && ipHost_B != NULL&&ipHost_C != NULL){
        for (int i=0; i<nxy; i++) {
            ipHost_A[i] = i;
            ipHost_B[i] = i + 1;
        }
        memset(ipHost_C, 0, stBytesCount);
    } else {
        printf("Fail to allocate host emmory!\n");
        exit(-1);
    }

    // 分配设备内存，并初始化
    int *ipDevice_A, *ipDevice_B, *ipDevice_C;
    cudaMalloc((int**)&ipDevice_A, stBytesCount);
    cudaMalloc((int**)&ipDevice_B, stBytesCount);
    cudaMalloc((int**)&ipDevice_C, stBytesCount);

    if (ipDevice_A != NULL && ipDevice_B != NULL && ipDevice_C !=NULL ) {
        cudaMemcpy(ipDevice_A, ipHost_A, stBytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(ipDevice_B, ipHost_B, stBytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(ipDevice_C, ipHost_C, stBytesCount, cudaMemcpyHostToDevice);
    } else {
        printf("fail to allocate memory\n");
        free(ipHost_A);
        free(ipHost_B);
        free(ipHost_C);
        exit(-1);
    }

    // 调用核函数在设备中进行计算
    dim3 block(4, 4);
    dim3 grid((nx + block.x -1) / block.x, (ny + block.y -1) / block.y); // (4, 2)
    // dim3 grid((iElemCount + block.x -1) / 32); // 不为整数则多一个线程块
    printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);

    addMatrix<<<grid, block>>>(ipDevice_A, ipDevice_B, ipDevice_C, nx, ny);
   
    ErrorCheck(cudaMemcpy(ipHost_C, ipDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);

    for (int i=0; i<10; i++) {
        printf("id=%d, matrix_A=%d, matrix_B=%d, result=%d\n", i+1, ipHost_A[i], ipHost_B[i], ipHost_C[i]);
    }

 
    // 释放内存
    free(ipHost_A);
    free(ipHost_B);
    free(ipHost_C);
    cudaFree(ipDevice_A);
    cudaFree(ipDevice_B);
    cudaFree(ipDevice_C);

    cudaDeviceReset();
    return 0;
}