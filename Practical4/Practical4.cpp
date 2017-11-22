// Practical4.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <xmmintrin.h>  
#include <iostream>
#include <chrono>
#include <time.h>
#include <random>

using namespace std;
using namespace chrono;

//Size of data to allocate - divide by 4 to get number of vectors
const unsigned int SIZE = static_cast<unsigned int>(pow(2, 24));
const unsigned int NUM_VECTORS = SIZE / 4;


void generate_data(float* data, unsigned int num_values)
{
	//random engine
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(static_cast<unsigned int>(millis.count()));

	//filling data
	for (unsigned int i = 0; i < num_values; ++i)
	{
		data[i] = e();
	}
}

void normalise_Vector(__m128* data, __m128* result, unsigned int num_vectors)
{
	for(unsigned int i = 0; i < num_vectors; ++i)
	{
		result[i] = _mm_mul_ps(data[i], data[i]);	//getting data^2

		//Calculate sum of components and store in all
		result[i].m128_f32[0] = result[i].m128_f32[1] = result[i].m128_f32[2] = result[i].m128_f32[3] =
			result[i].m128_f32[0] + result[i].m128_f32[1] + result[i].m128_f32[2] + result[i].m128_f32[3];
		//calculate recipricol square root of values
		//it is like doing 1.0f / sqrt(value)
		result[i] = _mm_rsqrt_ps(result[i]);

		//multiply result by original data --> no need to divide as we already have the recipricol
		result[i] = _mm_mul_ps(data[i], result[i]);
	}
}

void check_results(__m128* data, __m128* result)
{
	auto float_data = (float*)data;
	auto float_res = (float*)result;

	for (unsigned int i = 0; i < 100; ++i)
	{
		float l = 0.0f;
		//squre each component and add to 1
		for (unsigned int j = 0; j < 4; ++j)
		{
			l += powf(float_data[(i * 4) + j], 2.0f);
		}
		l = sqrtf(l);

		for (unsigned int j = 0; j < 4; j++)
		{
			cout << float_data[(i * 4) + j] / l << " : " << float_res[(i * 4) + j] << endl;
		}
	}
}

int main()
{
	/// ---------------------------------------------- INTRODUCTION ---------------------------------------------- ///

	////declaring single 128-bit value aligned to 16 bytes
	//__declspec(align(16)) __m128 x;							//values take up to 128-bits of memory
	//														//align(16) guarantees data aligned in memory at 16 byte block - or 128-bits

	////treating x as collection of 4 floats
	////or other combinations of values for 128-bits
	//x.m128_f32[0] = 10.0f;
	//x.m128_f32[1] = 20.0f;
	//x.m128_f32[2] = 30.0f;
	//x.m128_f32[3] = 40.0f;

	//cout << x.m128_f32[0] << endl;

	////Aligned memory is faster to access in blocks


	////create array of SIZE floats aligned to 4 bytes (size of a float)
	//float* data = (float*)_aligned_malloc(SIZE * sizeof(float), 4);			//malloc = memory allocation

	//cout << data[0] << endl;

	////Create and array of SIZE 128-bit values aligned to 16 bytes

	//__m128* big_data = (__m128*)_aligned_malloc(SIZE * sizeof(__m128), 16);

	//cout << big_data[0].m128_f32[0] << endl;

	////ALWAYS REMEMBER TO FREE MEMORY
	//_aligned_free(data);
	//_aligned_free(big_data);

	/// ---------------------------------------------- SIMD Operations ---------------------------------------------- ///

//	//aligning data to 16 bytes (128-bits)
//
//	auto data = (float*)_aligned_malloc(SIZE * sizeof(float), 16);
//
//	//initialise data
//
//	for (unsigned int i = 0; i < SIZE; ++i)
//	{
//		//set all values to 1.0f
//		data[i] = 1.0f;
//	}
//
//	//value to add to all values
//	auto value = _mm_set1_ps(4.0f);
//
//	//__m128 pointer to data
//	auto stream_data = (__m128*) data;
//
//	//timer
//	auto start = high_resolution_clock::now();
//
////Trying to mix up OMP and SIMD --> I got an improvement
////#pragma omp parallel for
//	for (int i = 0; i < NUM_VECTORS; ++i)
//	{
//		stream_data[i] = _mm_add_ps(stream_data[i], value);
//	}
//
//	auto end = high_resolution_clock::now();
//	auto total = duration_cast<microseconds>(end - start).count();
//	cout << "SIMD: " << total << " micros" << endl;
//
//	_aligned_free(data);
//
//	//Declare standard data
//	data = new float[SIZE];
//
//	for (unsigned int i = 0; i < SIZE; ++i)
//	{
//		//set all values to 1.0f
//		data[i] = 1.0f;
//	}
//
//	start = high_resolution_clock::now();
//
//	for (int i = 0; i < SIZE; ++i)
//	{
//		data[i] = data[i] + 4.0f;
//	}
//
//	end = high_resolution_clock::now();
//	total = duration_cast<microseconds>(end - start).count();
//	cout << "Non-SIMD: " << total << " micros" << endl;
//
//	delete[] data;

/// ---------------------------------------------- NORMALIZING A VECTOR ---------------------------------------------- ///



	return 0;
}

