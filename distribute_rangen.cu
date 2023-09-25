#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <random>
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
        printf("Error at %s:%d\n", __FILE__,__LINE__); \
        return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
        printf("Error at %s:%d\n", __FILE__,__LINE__);\
        return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::clog << "There is no device supporting CUDA." << std::endl;
  }

  size_t n = 100;
  size_t i;
  curandGenerator_t gen;
  float *devData, *hostData;

  hostData = (float *)calloc(n * deviceCount, sizeof(float));

  for (int dev=0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);

    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // set seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));

    // copy device memory to host
    CURAND_CALL(cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost));

    // clean up
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
  }

  for (i=0; i<n * deviceCount; ++i) {
      printf("%1.4f ", hostData[i]);
  }
  printf("\n");
  free(hostData);
  return EXIT_SUCCESS;
}
