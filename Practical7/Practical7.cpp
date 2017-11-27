// Practical7.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CL\cl.h"
#include <vector>

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


int main()
{
	cl_int status;
	vector<cl_platform_id> platforms;
	vector<cl_device_id> devices;
	cl_context context;
	cl_command_queue cmd_queue;

	initialise_opencl(platforms, devices, context, cmd_queue);

	//Free OpenCL resources
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);

    return 0;
}

