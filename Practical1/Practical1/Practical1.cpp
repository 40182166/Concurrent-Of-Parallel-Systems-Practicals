// Practical1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <thread>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>
#include <fstream>

using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

//const int num_threads = 100;



void task_one()
{
	cout << "Task 1 starting" << endl;
	cout << "Task 1 sleeping for 3 seconds" << endl;
	sleep_for(seconds(3));
	cout << "Task 1 awake again" << endl;
	cout << "Task 1 sleeping for 5 seconds" << endl;
	sleep_for(milliseconds(5000));
	cout << "Task 1 awake again" << endl;
	cout << "Task 1 ending" << endl;
}

void task_two()
{
	cout << "Task 2 starting" << endl;
	cout << "Task 2 sleeping for 2 seconds" << endl;
	sleep_for(microseconds(2000000));
	cout << "Task 2 awake again" << endl;
	cout << "Task 2 sleeping for 10 seconds" << endl;
	sleep_for(seconds(10));
	cout << "Task 2 awake again" << endl;
	cout << "Task 2 ending" << endl;
}

void task(int n, int val)
{
	cout << "Thread: " << n << " Random Value: " << val << endl;
}

void work()
{
	int n = 0;
	for (int i = 0; i < 1000000; i++)
	{
		n++;
	}
		

}

void monte_carlo_pi(unsigned int iterations)
{
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(millis.count());
	uniform_real_distribution<double> distribution(0.0, 1.0);

	unsigned int in_circle = 0;

	for (unsigned int i = 0; i < iterations; i++)
	{
		auto x = distribution(e);
		auto y = distribution(e);

		//Get length of vector with Pythagarous
		auto lenght = sqrt((x * x) + (y * y));

		if (lenght <= 1.0)
			in_circle++;
	}

	auto pi = (4.0 * in_circle) / static_cast<double>(iterations);
}

int random_number()
{
	auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	default_random_engine e(static_cast<int>(millis.count()));

	return e();
}

void get_min_max(vector<int> allNumbers, int *min, int *max)
{
	auto minMax = minmax_element(allNumbers.begin(), allNumbers.end());
	*min = *minMax.first;
	*max = *minMax.second;
}

//int get_max(vector<int> allNumbers)
//{
//	return  *max_element(allNumbers.begin(), allNumbers.end());
//}
//
//int get_min(vector<int> allNumbers)
//{
//	return  *min_element(allNumbers.begin(), allNumbers.end());
//}

int main()
{
	/*ofstream data("data.csv", ofstream::out);

	for (int i = 0; i < 100; i++)
	{
		auto start = system_clock::now();

		thread t(work);
		t.join();

		auto end = system_clock::now();
		auto total = end - start;
		data << duration_cast<milliseconds>(total).count() << endl;
	}

	data.close();*/

	////////////////////////////////////////// ------ MONTE CARLO ------ //////////////////////////////

	//ofstream data("montecarlo.csv", ofstream::out);

	//for (unsigned int num_threads = 1; num_threads <= 6; num_threads++)
	//{
	//	auto total_threads = static_cast<unsigned int>(pow(2.0, num_threads));
	//	cout << "Number of threads = " << total_threads << endl;
	//	data << "num_threads_ " << total_threads;

	//	for (unsigned int iters = 0; iters < 10; iters++)
	//	{
	//		auto start = system_clock::now();
	//		cout << iters << endl;
	//		vector<thread> threads;

	//		for (unsigned int n = 0; n < total_threads; n++)
	//		{
	//			threads.push_back(thread(monte_carlo_pi, static_cast<unsigned int>(pow(2.0, 24.0 - num_threads))));
	//		}

	//		for (auto &t : threads)
	//		{
	//			t.join();
	//		}
	//		
	//		auto end = system_clock::now();
	//		auto total = end - start;
	//		data << ", " << duration_cast<milliseconds>(total).count() << endl;
	//	}

	//	data << endl;
	//}
	//data.close();

	//////////////// Excercises ////////////////

	// 1

	ofstream data("excercise1.csv", ofstream::out);

	vector<int> allNumbers;

	//int max = 0;
	//int min = 0;

	for (unsigned int n = 0; n < 100; n++)
	{
		int rand = random_number();
		allNumbers.push_back(rand);
		//cout << rand << endl;
	}

	/*get_min_max(allNumbers, min, max);
	cout << "Max before: " << max << endl;
	cout << "Min before: " << min << endl;

	min = get_min(allNumbers);
	cout << "Min after: " << min << endl;
	max = get_max(allNumbers);
	cout << "Max after: " << max << endl;*/

	for (unsigned int num_threads = 1; num_threads <= 6; num_threads++)
		{
			auto total_threads = static_cast<unsigned int>(pow(2.0, num_threads));
			cout << "Number of threads = " << total_threads << endl;
			data << "num_threads_ " << total_threads;

			int mymin;
			int mymax;

			auto start = system_clock::now();
			vector<thread> threads;

			for (unsigned int n = 0; n < total_threads; n++)
			{
				threads.push_back(thread(get_min_max, allNumbers, &mymin, &mymax));
			}

			for (auto &t : threads)
			{
				t.join();
			}
			cout << "MIN = " << mymin << endl;
			cout << "MAX = " << mymax << endl;

			data << ", " << "MIN = "<< mymin << endl;
			data << ", " << "MAX = " << mymax << endl;
			auto end = system_clock::now();
			auto total = end - start;
			data << ", " << duration_cast<nanoseconds>(total).count() << endl;
			data << endl;

		}

		data.close();

	
    return 0;
}

