#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <hiprand_kernel.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>

// we could vary M & N to find the perf sweet spot
struct estimate_pi :
 public thrust::unary_function<unsigned int, float>
{
 __device__
 float operator()(unsigned int thread_id)
 {
 float sum = 0;
 unsigned int N = 10000; // samples per thread
 unsigned int seed = thread_id;
 printf("thread_id: %d\n ", seed);
 hiprandState s;
 // seed a random number generator
 hiprand_init(seed, 0, 0, &s);
 // take N samples in a quarter circle
 for(unsigned int i = 0; i < N; ++i)
 {
 // draw a sample from the unit square
 float x = hiprand_uniform(&s);
 float y = hiprand_uniform(&s);
 // measure distance from the origin
 float dist = sqrtf(x*x + y*y);
 // add 1.0f if (u0,u1) is inside the quarter circle
 if(dist <= 1.0f)
 sum += 1.0f;
 }
 // multiply by 4 to get the area of the whole circle
 sum *= 4.0f;
 // divide by N
 return sum / N;
 }
};
int main(void)
{
 // use 30K independent seeds
 int M = 30000;
 int device_count = 0;
 hipGetDeviceCount(&device_count);
 std::cout << device_count << std::endl;
 float estimate = thrust::transform_reduce(
 thrust::counting_iterator<int>(0),
 thrust::counting_iterator<int>(M),
 estimate_pi(),
 0.0f,
 thrust::plus<float>());
 estimate /= M;
 std::cout << std::setprecision(3);
 std::cout << "pi is approximately ";
 std::cout << estimate << std::endl;
 return 0;
}
