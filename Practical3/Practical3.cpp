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
#include <fstream>

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
	for (phase = 0; phase < n; phase++)
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

	//////////////// ------------ 3.4 Parallel Sort ------------ ////////////////

	ofstream results("parallel.csv", ofstream::out);

	for (unsigned int size = 8; size <= 16; size++)
	{
		results << pow(2, size) << ", ";
		for (unsigned int i = 0; i < 100; i++)
		{
			cout << "Generating " << i << " for " << pow(2, size) << " values" << endl;
			auto data = generate_values(static_cast<unsigned int>(pow(2, size)));

			cout << "Sorting" << endl;

			auto start = system_clock::now();
			parallel_sort(data);
			auto end = system_clock::now();

			auto total = duration_cast<milliseconds>(end - start).count();

			results << total << " , ";
		}
		results << endl;
	}
	results.close();


    return 0;
}

