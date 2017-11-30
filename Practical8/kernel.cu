
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

using namespace std;

__global__ void vecadd(const int* A, const int* B, int* C)
{
	//get block index
	unsigned int block_idx = blockIdx.x;

	//get thread index
	unsigned int thread_idx = threadIdx.x;

	//get number of threads per block
	unsigned int block_dim = blockDim.x;

	//get the thread's unique ID - (block_idx * block_dim) + thread_idx
	unsigned int idx = (block_idx * block_dim) + thread_idx;

	//add corresponding locations of A and B and store in C - same as opencl
	C[idx] = A[idx] + B[idx];
}

void cuda_info()
{
	//get cuda device
	int device;
	cudaGetDevice(&device);

	//get cuda device properties
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);

	//display properties
	cout << "Name: " << properties.name << endl;
	cout << "CUDA Capability: " << properties.major << "." << properties.minor <<endl;
	cout << "Cores: " << properties.multiProcessorCount << endl;
	cout << "Memory: " << properties.totalGlobalMem / (1024 * 1024) << " MB" << endl;
	cout << "Clock freq: " << properties.clockRate / 1000 << " MHz" << endl;
	cout << "***************************************************************" << endl << endl;

}

int main()
{

	//initialise CUDA
	cudaSetDevice(0);
	cuda_info();

	const unsigned int ELEMENTS = 2048;
	//create host memory
	auto data_size = sizeof(int) * ELEMENTS;

	vector<int> A(ELEMENTS);
	vector<int> B(ELEMENTS);
	vector<int> C(ELEMENTS);

	//initialise data
	for (unsigned int i = 0; i < ELEMENTS; ++i)
	{
		A[i] = B[i] = i;
	}

	//declare buffers
	int *buffer_A, *buffer_B, *buffer_C;

	//initialise buffers
	cudaMalloc((void**)&buffer_A, data_size);
	cudaMalloc((void**)&buffer_B, data_size);
	cudaMalloc((void**)&buffer_C, data_size);

	//copy memory from host to device
	cudaMemcpy(buffer_A, &A[0], data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(buffer_B, &B[0], data_size, cudaMemcpyHostToDevice);

	//run kernel with one thread for each element
	//first value is number of blocks, second is threads per block
	//max 1024 threads per block
	vecadd<<<ELEMENTS / 1024, 1024 >>>(buffer_A, buffer_B, buffer_C);

	//wait for kernel to complete
	cudaDeviceSynchronize();

	//read output buffer back to the host
	cudaMemcpy(&C[0], buffer_C, data_size, cudaMemcpyDeviceToHost);

	//clean up resources
	cudaFree(buffer_A);
	cudaFree(buffer_B);
	cudaFree(buffer_C);

    return 0;
}
