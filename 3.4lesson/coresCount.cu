#include <stdio.h>
#include "../tools/common.cuh"


int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
    {
    case 2:
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp *32;
        /* code */
      break;
    case 3:
      cores = mp * 192;
    case 5:
      cores = mp * 128;
      break;
    case 6:
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp *64;
      else printf("Unknown device type\n");
      break;
    case 7:
      if ((devProp.minor == 0) ||(devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 8:
      if (devProp.minor == 0) cores = mp *64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
    case 9:
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
    default:
      printf("unknown device type\n");
      break;
    }
    return cores;
}

int main() {
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);
    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop,device_id), __FILE__, __LINE__);
    printf("COMPUTE CORES IS %d\n", getSPcores(prop));
    return 0;
}