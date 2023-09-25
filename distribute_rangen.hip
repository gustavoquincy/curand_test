#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <random>
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=hipSuccess) { \
        printf("Error at %s:%d\n", __FILE__,__LINE__); \
        return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=HIPRAND_STATUS_SUCCESS) { \
        printf("Error at %s:%d\n", __FILE__,__LINE__);\
        return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::clog << "There is no device supporting CUDA." << std::endl;
  }

  size_t n = 100;
  size_t i;
  hiprandGenerator_t gen;
  float *devData, *hostData;

  hostData = (float *)calloc(n * deviceCount, sizeof(float));

  for (int dev=0; dev < deviceCount; ++dev) {
    hipSetDevice(dev);

    CUDA_CALL(hipMalloc((void **)&devData, n*sizeof(float)));

    CURAND_CALL(hiprandCreateGenerator(&gen, HIPRAND_RNG_PSEUDO_DEFAULT));

    // set seed
    CURAND_CALL(hiprandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    // generate n floats on device
    CURAND_CALL(hiprandGenerateUniform(gen, devData, n));

    // copy device memory to host
    CURAND_CALL(hipMemcpy(hostData, devData, n * sizeof(float), hipMemcpyDeviceToHost));

    // clean up
    CURAND_CALL(hiprandDestroyGenerator(gen));
    CUDA_CALL(hipFree(devData));
  }

  for (i=0; i<n * deviceCount; ++i) {
      printf("%1.4f ", hostData[i]);
  }
  printf("\n");


  free(hostData);
  return EXIT_SUCCESS;
}
