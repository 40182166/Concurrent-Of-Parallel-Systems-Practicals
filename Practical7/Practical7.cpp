// Practical7.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CL\cl.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <array>

using namespace std;


void initialise_opencl(vector<cl_platform_id> &platforms, vector<cl_device_id> &devices, cl_context &context, cl_command_queue &cmd_queue)
{
	//status of opencl calls
	cl_int status;

	//getting number of platforms

	cl_uint num_platforms;
	status = clGetPlatformIDs(0, nullptr, &num_platforms);
	//resize vector to store platforms
	platforms.resize(num_platforms);
	
	//fill in platform vector
	status = clGetPlatformIDs(num_platforms, &platforms[0], nullptr);

	// Assume platform 0 is the one we want to use
	// Get devices for platform 0
	cl_uint num_devices;
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
	//resize vector to store devices
	devices.resize(num_devices);

	//fill in devices vector
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, &devices[0], nullptr);

	//create a context
	context = clCreateContext(nullptr, num_devices, &devices[0], nullptr, nullptr, &status);

	//create a command queue
	cmd_queue = clCreateCommandQueue(context, devices[0], 0, &status);
}

void print_opencl_info(vector<cl_device_id> &devices)
{
	// buffers device name and vendor
	char device_name[1024], vendor[1024];

	//declare other necessary variables
	cl_uint num_cores;
	cl_long memory;
	cl_uint clock_freq;
	cl_bool available;

	for (auto &d : devices)
	{
		clGetDeviceInfo(d, CL_DEVICE_NAME, 1024, device_name, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_VENDOR, 1024, vendor, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cores, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_long), &memory, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock_freq, nullptr);
		clGetDeviceInfo(d, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, nullptr);

		// Print info
		cout << "Device: " << device_name << endl;
		cout << "Vendor: " << vendor << endl;
		cout << "Cores: " << num_cores << endl;
		cout << "Memory: " << memory / (1024 * 1024) << " MB" << endl;
		cout << "Clock freq: " << clock_freq << "MHz" << endl;
		cout << "Available: " << available << endl;
		cout << "*************************************** " << endl << endl;
	}
}

//loading kernel

cl_program load_program(const string &filename, cl_context &context, cl_device_id &device, cl_int num_devices)
{
	cl_int status;

	//create and compile program
	//read in kernel file
	ifstream input(filename, ifstream::in);
	stringstream buffer;
	buffer << input.rdbuf();

	//get the character array of the file contents
	auto file_contents = buffer.str();
	auto char_contents = file_contents.c_str();

	//create program object
	auto program = clCreateProgramWithSource(context, 1, &char_contents, nullptr, &status);
	// compile/build program
	status = clBuildProgram(program, num_devices, &device, nullptr, nullptr, nullptr);

	if (status != CL_SUCCESS)
	{
		//error building
		size_t length;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
		char* log = new char[length];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log, &length);

		cout << log << endl;
		delete[] log;
	}

	return program;
}

int main()
{
	cl_int status;
	vector<cl_platform_id> platforms;
	vector<cl_device_id> devices;
	cl_context context;
	cl_command_queue cmd_queue;
	initialise_opencl(platforms, devices, context, cmd_queue);
	print_opencl_info(devices);


	const unsigned int elements = 2048;
	const unsigned int data_size = sizeof(int) * elements;

	//host data - stored in main memory
	array<int, elements> A;
	array<int, elements> B;
	array<int, elements> C;

	//initialise input data
	for (unsigned int i = 0; i < elements; ++i)
	{
		A[i] = B[i] = i;
	}

	//create device buffers - stored on GPU
	cl_mem buffer_A;	//input array on device
	cl_mem buffer_B;	//input array on device
	cl_mem buffer_C;	//output array on device

						//allocate buffer size
	buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, nullptr, &status);
	buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, nullptr, &status);
	buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, nullptr, &status);

	status = clEnqueueWriteBuffer(cmd_queue, buffer_A, CL_FALSE, 0, data_size, A.data(), 0, nullptr, nullptr);
	status = clEnqueueWriteBuffer(cmd_queue, buffer_B, CL_FALSE, 0, data_size, B.data(), 0, nullptr, nullptr);

	//load program
	auto program = load_program("kernel.cl", context, devices[0], devices.size());

	//create the kernel
	auto kernel = clCreateKernel(program, "vecadd", &status);

	//set the kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_B);


	//Free OpenCL resources
	clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_B);
	clReleaseMemObject(buffer_C);
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

    return 0;
}

