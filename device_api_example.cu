#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/compute/api.h>
#include <arrow/util/type_fwd.h>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { printf("Error at %s:%d\n", __FILE__,__LINE__); return EXIT_FAILURE; }} while(0)
#pragma GCC diagnostic ignored "-Wunused-result"

__global__ __launch_bounds__(1024) void setup_kernel(curandState *state, int seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets device index seed, a different sequence number, no offset */
  curand_init(seed, id, 0, &state[id]);
}

__global__ __launch_bounds__(1024) void generate_uniform_kernel(curandState *state, double_t *result, double_t *result_2,  int sampleSize, int dev, int offset)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(dev, id, offset, &state[id]);
  if (id < sampleSize) result[id] = curand_uniform_double(&state[id]);
	if (id < sampleSize) result_2[id] = curand_uniform_double(&state[id]);
}

arrow::Status write_csv_from_thrust_device(double_t *in, double_t *in_2,  int64_t size, std::string output_name) {
	arrow::DoubleBuilder doublebuilder;
	ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(in, size));
	std::shared_ptr<arrow::Array> random_number;
	ARROW_ASSIGN_OR_RAISE(random_number, doublebuilder.Finish());
	std::shared_ptr<arrow::ChunkedArray> chunks= std::make_shared<arrow::ChunkedArray>(random_number);
	ARROW_RETURN_NOT_OK(doublebuilder.AppendValues(in_2, size));
	std::shared_ptr<arrow::Array> random_number_2;
	ARROW_ASSIGN_OR_RAISE(random_number_2, doublebuilder.Finish());
	std::shared_ptr<arrow::ChunkedArray> chunks_2 = std::make_shared<arrow::ChunkedArray>(random_number_2);
	std::shared_ptr<arrow::Field> field_1, field_2;
	std::shared_ptr<arrow::Schema> schema;
	field_1 = arrow::field("devResults", arrow::float64());
	field_2 = arrow::field("devResults_2", arrow::float64());
	schema = arrow::schema({field_1, field_2});
	std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {chunks, chunks_2});
	std::shared_ptr<arrow::io::FileOutputStream> outfile;
	ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(output_name));
	ARROW_ASSIGN_OR_RAISE(auto csv_writer, arrow::csv::MakeCSVWriter(outfile, table->schema()));
	ARROW_RETURN_NOT_OK(csv_writer->WriteTable(*table));
	ARROW_RETURN_NOT_OK(csv_writer->Close());

	return arrow::Status::OK();
}	

int main(int argc, char *argv[1])
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  const unsigned int threadPerBlock = 10;//1024;
  const unsigned int blockCount = 1;//207520;
  const unsigned int totalThreads = threadPerBlock * blockCount;
  
  curandState *devStates;
  int sampleSize = 10;//212500000;
  double_t *devResults, *hostResults, *devResults_2, *hostResults_2;
  hostResults = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
	hostResults_2 = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
  #pragma omp parallel for num_threads(4) private(devResults, devResults_2, devStates) shared(sampleSize, totalThreads, blockCount, threadPerBlock, deviceCount)
  for (int dev=0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaMalloc((void **)&devResults, sampleSize * sizeof(double_t));
    cudaMemset(devResults, 0, sampleSize * sizeof(double_t));
    cudaMalloc((void **)&devResults_2, sampleSize * sizeof(double_t));
		cudaMemset(devResults_2, 0, sampleSize * sizeof(double_t));
		cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
		generate_uniform_kernel<<<blockCount, threadPerBlock>>>(devStates, devResults, devResults_2, sampleSize, dev, 0);
		cudaMemcpy(hostResults + dev * sampleSize, devResults, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostResults_2 + dev * sampleSize, devResults_2, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
   	cudaFree(devResults);
		cudaFree(devResults_2);
  }
  thrust::device_vector<double_t> device(hostResults, hostResults +  sampleSize * deviceCount);
  thrust::device_vector<double_t> device_2(hostResults_2, hostResults_2 + sampleSize * deviceCount);
	free(hostResults);
	free(hostResults_2);
	/*
	curandState *devStates_2;	
	double_t *devResults_2, *hostResults_2;	
	hostResults_2 = (double_t *)calloc(sampleSize * deviceCount, sizeof(double_t));
	#pragma omp parallel for num_threads(4) private(devResults_2, devStates_2) shared(sampleSize, totalThreads, blockCount, threadPerBlock)
	for (int dev=0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
	  cudaMalloc((void **)&devResults_2, sampleSize * sizeof(double_t));
		cudaMemset(devResults_2, 0, sampleSize * sizeof(double_t));
		cudaMalloc((void **)&devStates_2, totalThreads * sizeof(curandState));
		setup_kernel<<<blockCount, threadPerBlock>>>(devStates_2, dev);
		generate_uniform_kernel<<<blockCount, threadPerBlock>>>(devStates_2, devResults_2, sampleSize);
		cudaMemcpy(hostResults_2 + dev * sampleSize, devResults_2, sampleSize * sizeof(double_t), cudaMemcpyDeviceToHost);
		cudaFree(devResults_2);
	}
	thrust::device_vector<double_t> device_2(hostResults_2, hostResults_2 + sampleSize * deviceCount);
	free(hostResults_2);
	*/
	double_t *raw = thrust::raw_pointer_cast(device.data());
	int64_t size = device.size();
	device.clear();
	
	double_t *raw_2 = thrust::raw_pointer_cast(device_2.data());
	int64_t size_2 = device_2.size();
	device_2.clear();

	arrow::Status status = write_csv_from_thrust_device(raw, raw_2, size, "output.csv");
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return EXIT_FAILURE;
	}
	/*
	status = write_csv_from_thrust_device(raw_2, size_2, "output_2.csv");
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return EXIT_FAILURE;
	}
	*/
	return EXIT_SUCCESS;
}
