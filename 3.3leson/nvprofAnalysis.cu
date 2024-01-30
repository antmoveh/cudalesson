#include <stdio.h>
#include "../tools/common.cuh"


#define NUM_REPEATS 10

// 设备函数,只能被核函数调用
__device__ float add(const float x, const float y) {
    return x + y;
}

// 核函数，并行计算
__global__ void addFromGPU(float *A, float *B, float *C, const int N) {
    
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    // 元素数量和线程数相同
    // C[id] = A[id] + B[id];
    // 元素数量和线程数不同
    if (id>N) return;
    C[id] = add(A[id], B[id]);
}

void initialData(float *addr, int elemCount) {
    for (int i=0; i<elemCount; i++){
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}


int main(void) {

    //  设置运行GPU
    setGPU();

    // 分配主机内存并初始化
    int iElemCount = 512;
    size_t stBytesCount = iElemCount * sizeof(float);


    float *fpHost_A, *fpHost_B, *fpHost_C;

    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);

    if (fpHost_A !=NULL && fpHost_B !=NULL&&fpHost_C!=NULL){
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    } else {
        printf("Fail to allocate host emmory!\n");
        exit(-1);
    }

    // 分配设备内存，并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float**)&fpDevice_A, stBytesCount);
    cudaMalloc((float**)&fpDevice_B, stBytesCount);
    cudaMalloc((float**)&fpDevice_C, stBytesCount);

    if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C !=NULL ) {
        cudaMemset(fpDevice_A, 0, stBytesCount);
        cudaMemset(fpDevice_B, 0, stBytesCount);
        cudaMemset(fpDevice_C, 0, stBytesCount);
    } else {
        printf("fail to allocate memory\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpDevice_C);
        exit(-1);
    }

    // 初始化主机中的数据
    srand(666);
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    // 将主机中的数据复制到显卡
    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // 调用核函数在设备中进行计算
    dim3 block(32);
    dim3 grid(iElemCount / 32); // 512/32=16
    // dim3 grid((iElemCount + block.x -1) / 32); // 不为整数则多一个线程块


    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);


    // Loop to print the values of fpHost_A, fpHost_B, and fpHost_C
    ErrorCheck(cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
 
    // 释放内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();
    return 0;
}