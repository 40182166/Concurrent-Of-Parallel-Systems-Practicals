
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

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


}

int main()
{

	//initialise CUDA
	cudaSetDevice(0);
	cuda_info();
    return 0;
}
