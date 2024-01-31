//---------- x.cu ----------
#include <stdio.h>
#include "y.h"

__global__ void foo (void) {

  __shared__ int a[N];
  a[threadIdx.x] = threadIdx.x;

  __syncthreads();

  g[threadIdx.x] = a[blockDim.x - threadIdx.x - 1];

  bar();
}

int main (void) {
  unsigned int i;
  int *dg, hg[N];
  int sum = 0;

  foo<<<1, N>>>();

  if(cudaGetSymbolAddress((void**)&dg, g)){
      printf("couldn't get the symbol addr\n");
      return 1;
  }
  if(cudaMemcpy(hg, dg, N * sizeof(int), cudaMemcpyDeviceToHost)){
      printf("couldn't memcpy\n");
      return 1;
  }

  for (i = 0; i < N; i++) {
    sum += hg[i];
  }
  if (sum == 36) {
    printf("PASSED\n");
  } else {
    printf("FAILED (%d)\n", sum);
  }

  return 0;
}

/*

// device code指的是CUDA相关代码，host object指的是c++代码编译出来的产物
// 将x.cu和y.cu中的device code分别嵌入到其对应的host object中，即x.o和y.o
➜  nvcc --gpu-architecture=sm_50 --device-c x.cu y.cu
// 使用device-link将x.o和y.o中的device code链接在一起，得到link.o
➜  nvcc --gpu-architecture=sm_50 --device-link x.o y.o --output-file a_dlink.o
// 将链接后的link.o和其他host object链接在一起，得到最终产物
➜  g++ x.o y.o a_dlink.o -L<path> -lcudart // 这里<path>替换成你libcudart.so对应路径

*/