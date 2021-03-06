// Practical3.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include <cmath>
#include <math.h>
#include <fstream>
#define _USE_MATH_DEFINES

using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

const int THREADS = 10;

void hello()
{
	//thread number
	auto my_rank = omp_get_thread_num();

	//threads in operation
	auto thread_count = omp_get_num_threads();

	cout << "This is thread number " << my_rank << " of " << thread_count << endl;
}

vector<unsigned int> generate_values(unsigned int size)
{
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(static_cast<unsigned int>(millis.count()));

	vector<unsigned int> data;

	for (unsigned int i = 0; i < size; i++)
	{
		data.push_back(e());
	}

	return data;
}

void bubble_sort(vector<unsigned int>& data)
{
		for (unsigned int i = 0; i < data.size() - 1; i++)
		{
			if (data[i] < data[i + 1])
			{
				auto tmp = data[i];
				data[i] = data[i + 1];
				data[i + 1] = tmp;
			}
		}
}

void parallel_sort(vector<unsigned int>& values)
{
	auto num_threads = thread::hardware_concurrency();
	auto n = values.size();

	int i, tmp, phase;

#pragma omp parallel num_threads(num_threads) default(none) shared(values, n) private(i, tmp, phase)
	for (phase = 0; phase < n; ++phase)
	{
		if (phase % 2 == 0)
		{
#pragma omp for
			for (i = 1; i < n; i += 2)
			{
				if (values[i - 1] > values[i])
				{
					tmp = values[i - 1];
					values[i - 1] = values[i];
					values[i] = tmp;
				}
			}
		}
		else
		{
#pragma omp for
			for (i = 1; i < n; i += 2)
			{
				if (values[i] > values[i + 1])
				{
					tmp = values[i + 1];
					values[i + 1] = values[i];
					values[i] = tmp;
				}
			}
		}
	}
}

void trap(function<double(double)> f, double start, double end, unsigned int iterations, shared_ptr<double> p)
{
	auto my_rank = omp_get_thread_num();
	auto thread_count = omp_get_num_threads();
	auto slice_size = (end - start) / iterations;
	auto iterations_thread = iterations / thread_count;
	auto local_start = start + ((my_rank * iterations_thread) * slice_size);
	auto local_end = local_start + iterations_thread * slice_size;
	auto my_result = (f(local_start) + f(local_end)) / 2.0;

	double x;

	for (unsigned int i = 0; i <= iterations_thread - 1; i++)
	{
		x = local_start + i * slice_size;
		my_result += f(x);
	}

	my_result *= slice_size;

#pragma omp critical
	*p += my_result;
}

double f(unsigned int i)
{
	auto start = i * (i + 1) / 2;
	auto end = start + i;

	auto result = 0.0;

	for (auto j = start; j <= end; j++)
	{
		result += sin(j);
	}

	return result;
}

int main()
{

	//////////////// ------------ 3.1 First OpenMP App ------------ ////////////////

	//#pragma omp parallel num_threads(THREADS) 
	//hello();

	//////////////// ------------ 3.2 parallel for ------------ ////////////////

	//auto num_threads = thread::hardware_concurrency();
	//const int n = static_cast<int>(pow(2, 30));
	//double factor = 0.0;
	//double pi = 0.0;

	//#pragma omp parallel for num_threads(num_threads) reduction(+:pi) private(factor)

	//for (int k = 0; k < n; k++)
	//{
	//	if (k % 2 == 0)
	//	{
	//		factor = 1.0;
	//	}
	//	else
	//	{
	//		factor = -1.0;
	//	}

	//	pi += factor / (2.0 * k + 1);
	//}

	//pi *= 4.0;

	////for more precision
	//cout.precision(numeric_limits<double>::digits10);
	//cout << "pi = " << pi << endl;

	//////////////// ------------ 3.3 Bubble Sort ------------ ////////////////

	//ofstream results("bubble.csv", ofstream::out);

	//for (unsigned int size = 8; size <= 16; size++)
	//{
	//	results << pow(2, size) << ", ";
	//	for (unsigned int i = 0; i < 100; i++)
	//	{
	//		cout << "Generating " << i << " for " << pow(2, size) << " values" << endl;
	//		auto data = generate_values(static_cast<unsigned int>(pow(2, size)));

	//		cout << "Sorting" << endl;

	//		auto start = system_clock::now();
	//		bubble_sort(data);
	//		auto end = system_clock::now();

	//		auto total = duration_cast<milliseconds>(end - start).count();

	//		results << total << " , ";
	//	}
	//	results << endl;
	//}
	//results.close();

	///////////////   TRYAL BUBBLE SORT   ///////////////////

	//vector<unsigned int> data;

	//data.push_back(5);
	//data.push_back(234);
	//data.push_back(543);
	//data.push_back(12);
	//data.push_back(2);
	//data.push_back(55);
	//data.push_back(9);

	//for (auto &d : data)
	//{
	//	cout << d << endl;
	//}

	//bubble_sort(data);

	//for (auto &d : data)
	//{
	//	cout << d << endl;
	//}

	//////////////// ------------ 3.4 Parallel Sort ------------ //////////////// DOESN'T WORK

	//ofstream results("parallel.csv", ofstream::out);

	//for (unsigned int size = 8; size <= 16; size++)
	//{
	//	results << pow(2, size) << ", ";
	//	for (unsigned int i = 0; i < 100; i++)
	//	{
	//		cout << "Generating " << i << " for " << pow(2, size) << " values" << endl;
	//		auto data = generate_values(static_cast<unsigned int>(pow(4, size)));

	//		cout << "Sorting" << endl;

	//		auto start = system_clock::now();
	//		parallel_sort(data);
	//		auto end = system_clock::now();

	//		auto total = duration_cast<milliseconds>(end - start).count();

	//		results << total << " , ";
	//	}
	//	results << endl;
	//}
	//results.close();


	//////////////// ------------ 3.5 Trapezoidal Rule ------------ ////////////////

//	auto result = make_shared<double>(0.0);
//	auto start = 0.0;
//	auto end = 3.14159265358;
//
//	unsigned int trapezoids = static_cast<unsigned int>(pow(2, 24));
//	auto thread_count = thread::hardware_concurrency();
//
//	auto f = [](double x) { return cos(x); };
//
//#pragma omp parallel num_threads(thread_count) 
//	trap(f, start, end, trapezoids, result);
//	cout << "Using " << trapezoids << " trapezoids. ";
//	cout << "Estimated integral of function " << start << " to " << end << " = " << *result << endl;

	//////////////// ------------ 3.6 Scheduling ------------ ////////////////

	auto thread_count = thread::hardware_concurrency();
	int n = static_cast<int>(pow(2, 14));
	auto start = system_clock::now();
	auto sum = 0.0;

#pragma omp parallel for num_threads(thread_count) reduction(+:sum) schedule(static, 1)
	for (auto i = 0; i <= n; i++)
	{
		sum += f(i);
	}

	auto end = system_clock::now();

	auto total = duration_cast<milliseconds>(end - start).count();

	cout << "total time: " << total << "ms" << endl;

    return 0;
}

