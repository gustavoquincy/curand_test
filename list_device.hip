#include <iostream>
#include <stdio.h>
#include <hiprand_kernel.h>
void list_devices(void)
{
  int deviceCount;
  hipGetDeviceCount(&deviceCount);
  if(deviceCount == 0)
  {
    std::cout << "There is no device supporting CUDA" << std::endl;
  }

  int selected_device;
  hipGetDevice(&selected_device);

  for (int dev = 0; dev < deviceCount; ++dev)
  {
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);

    if(dev == 0)
    {
      if(deviceProp.major == 9999 && deviceProp.minor == 9999)
        std::cout << "There is no device supporting CUDA." << std::endl;
      else if(deviceCount == 1)
        std::cout << "There is 1 device supporting CUDA" << std:: endl;
      else
        std::cout << "There are " << deviceCount <<  " devices supporting CUDA" << std:: endl;
    }

    std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"";
    if(dev == selected_device)
      std::cout << "  [SELECTED]";
    std::cout << std::endl;

    std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
    std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
    std::cout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    std::cout << "  Multiple processors count:                     " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  Shared memory:                                 " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
  }
  std::cout << std::endl;
}

int main(void) {

  list_devices();
  return 0;
}
