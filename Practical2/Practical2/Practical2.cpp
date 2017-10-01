// Practical2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "guarded1.h"
#include "threadsafe_stack.h"
#include <memory>
#include <thread>
#include <vector>
#include <random>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <future>
#include "FreeImage.h"

using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

mutex mut;

const unsigned int NUM_ITERATIONS = 1000000;
const unsigned int NUM_THREADS = 4;

const unsigned int max_iterations = 1000;
const unsigned int dim = 8192;	//image dimension in pixel

const double xmin = -2.1;
const double xmax = 1.0;
const double ymin = -1.3;
const double ymax = 1.3;

const double integral_x = (xmax - xmin) / static_cast<double>(dim);
const double integral_y = (ymax - ymin) / static_cast<double>(dim);


vector<double> mandelbrot(unsigned int start_y, unsigned int end_y)
{
	double x, y, x1, y1, xx = 0.0;
	unsigned int loop_count = 0;

	vector<double> results;

	y = ymin + (start_y * integral_y);

	for (unsigned int y_coord = start_y; y_coord < end_y; y_coord++)
	{
		x = xmin;
		for (unsigned int x_coord = 0; x_coord < dim; x_coord++)
		{
			x1 = 0.0, y1 = 0.0;
			loop_count = 0;
			while (loop_count < max_iterations && sqrt((x1 * x1) + (y1 * y1)) < 2.0)
			{
				loop_count++;
				xx = (x1 * x1) - (y1 * y1) + x;
				y1 = 2 * x1 * y1 + y;
				x1 = xx;
			}
			auto val = static_cast<double>(loop_count) / static_cast<double>(max_iterations);
			results.push_back(val);
			x += integral_x;
		}
		y += integral_y;
	}
	return results;
}



unsigned int find_max(const vector <unsigned int> &data, unsigned int start, unsigned int end)
{
	unsigned int max = 0;

	for (unsigned int i = start; i < end; i++)
	{
		if (data.at(i) > max)
		{
			max = data.at(i);
		}
	}

	return max;
}


void increment(shared_ptr<int> value)
{
	for (unsigned int i = 0; i < 1000000; i++)
	{
		lock_guard<mutex> lock(mut);
		*value = *value + 1;

	}
}

void incrementAtomic(shared_ptr<atomic<int>> value)
{
	for (unsigned int i = 0; i < 1000000; i++)
	{
		(*value)++;
	}
}

void taskAtomic(unsigned int id, shared_ptr<atomic_flag> flag)
{
	for (unsigned int i = 0; i < 10; i++)
	{
		while (flag->test_and_set());
		
		cout << "Thread " << id << " running " << i << endl;
	
		this_thread::sleep_for(seconds(1));
		flag->clear();
	}
}

void task(shared_ptr<guarded1> g)
{
	for (unsigned int i = 0; i < NUM_ITERATIONS; i++)
	{
		g->increment();
	}
}

void task_1(condition_variable &condition)
{
	cout << "Task 1 sleeping for 3 seconds" << endl;
	sleep_for(seconds(3));
	cout << "Task 1 notifying waiting thread" << endl;
	condition.notify_one();
	cout << "Task 1 waiting for notification" << endl;
	condition.wait(unique_lock<mutex>(mut));
	cout << "Task 1 notified" << endl;
	cout << "Task 1 sleeping for 3 seconds" << endl;
	sleep_for(seconds(3));
	cout << "Task 1 notifying waiting thread" << endl;
	condition.notify_one();
	cout << "Task 1 waiting 3 seconds for notification" << endl;
	if (condition.wait_for(unique_lock<mutex>(mut), seconds(3)) == cv_status::timeout)
	{
		cout << "Task 1 notified before 3 seconds" << endl;
	}
	else
	{
		cout << "Task 1 got tired waiting" << endl;
	}
	cout << "Task 1 finished" << endl;
}

void task_2(condition_variable &condition)
{
	cout << "Task 2 waiting for notification" << endl;
	condition.wait(unique_lock<mutex>(mut));
	cout << "Task 2 notified" << endl;
	cout << "Task 2 sleeping for 5 seconds" << endl;
	sleep_for(seconds(5));
	cout << "Task 2 notifying waiting thread" << endl;
	condition.notify_one();
	cout << "Task 2 waiting 5 seconds for notification" << endl;
	if (condition.wait_for(unique_lock<mutex>(mut), seconds(5)) == cv_status::timeout)
	{
		cout << "Task 2 notified before 5 seconds" << endl;
	}
	else
	{
		cout << "Task 2 got tired waiting" << endl;
	}
	cout << "Task 2 sleeping for 5 seconds" << endl;
	sleep_for(seconds(5));
	cout << "Task 1 waiting for notification" << endl;
	condition.wait(unique_lock<mutex>(mut));

	cout << "Task 1 notifying waiting thread" << endl;
	condition.notify_one();
	
	cout << "Task 1 finished" << endl;
}

void pusher(shared_ptr<threadsafe_stack<unsigned int>> stack)
{
	for (unsigned int i = 0; i < 1000000; i++)
	{
		stack->push(i);

		//give priority to another thread
		this_thread::yield();
	}
}

void popper(shared_ptr<threadsafe_stack<unsigned int>> stack)
{
	unsigned int count = 0;
	while (count < 1000000)
	{
		try
		{
			auto val = stack->pop();
			++count;
		}
		catch (exception e)
		{
			cout << e.what() << endl;
		}
	}
}

int main()
{

	FreeImage_Initialise;

	///////////////////////////////////////////////// ---- MUTEX ---- /////////////////////////////////////////////////
	//auto value = make_shared<int>(0);
	//auto num_threads = thread::hardware_concurrency();
	//vector<thread> threads;
	//for (unsigned int i = 0; i < num_threads; i++)
	//{
	//	threads.push_back(thread(increment, value));
	//}

	//for (auto &t : threads)
	//{
	//	t.join();
	//}

	//cout << "Value = " << *value << endl;

	///////////////////////////////////////////////// ---- CONDITION VARIABLE ---- /////////////////////////////////////////////////

	//condition_variable condition;

	//thread t1(task_1, ref(condition));
	//thread t2(task_2, ref(condition));

	//t1.join();
	//t2.join();

	///////////////////////////////////////////////// ---- GUARDED OBJECTS ---- /////////////////////////////////////////////////

	/*auto g = make_shared<guarded1>();
	
	vector<thread> threads;

	for (unsigned int i = 0; i < NUM_THREADS; i++)
		threads.push_back(thread(task, g));

	for (auto &t : threads)
		t.join();

	cout << "Value = " << g->get_value() << endl;*/

	///////////////////////////////////////////////// ---- THREAD SAFE DATA STRUCTURES ---- /////////////////////////////////////////////////

	//auto stack = make_shared<threadsafe_stack<unsigned int>>();

	//thread t1(popper, stack);
	//thread t2(pusher, stack);

	//t1.join();
	//t2.join();

	//cout << "Stack empty = " << stack->empty() << endl;


	///////////////////////////////////////////////// ---- ATOMICS ---- /////////////////////////////////////////////////

	//auto value = make_shared<atomic<int>>();
	//auto num_threads = thread::hardware_concurrency();
	//vector<thread> threads;
	//for (unsigned int i = 0; i < num_threads; i++)
	//{
	//	threads.push_back(thread(incrementAtomic, value));
	//}

	//for (auto &t : threads)
	//{
	//	t.join();
	//}

	//cout << "Value = " << *value << endl;

	///////////////////////////////////////////////// ---- ATOMIC_FLAG ---- /////////////////////////////////////////////////

    
	//auto flag = make_shared<atomic_flag>();

	//auto num_threads = thread::hardware_concurrency();
	//vector<thread> threads;
	//for (unsigned int i = 0; i < num_threads; i++)
	//{
	//	threads.push_back(thread(taskAtomic,i, flag));
	//}

	//for (auto &t : threads)
	//{
	//	t.join();
	//}

	///////////////////////////////////////////////// ---- FUTURES ---- /////////////////////////////////////////////////

	//auto num_threads = thread::hardware_concurrency();

	//vector<unsigned int> values;
	//auto millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	//default_random_engine e(static_cast<unsigned int>(millis.count()));

	//for (unsigned int i = 0; i < pow(2, 24); i++)
	//	values.push_back(e());

	//vector<future<unsigned int>> futures;
	//auto range = static_cast<unsigned int>(pow(2, 24) / num_threads);
	//
	//for (unsigned int i = 0; i < num_threads - 1; i++)
	//	futures.push_back(async(find_max, ref(values), i * range, (i + 1) * range));
	//
	//auto max = find_max(values, (num_threads - 1) * range, num_threads * range);

	//for (auto &f : futures)
	//{
	//	auto result = f.get();
	//	if (result > max)
	//		max = result;
	//}

	//cout << "Maximum value found: " << max << endl;

	///////////////////////////////////////////////// ---- FRACTALS ---- /////////////////////////////////////////////////

	auto num_threads = thread::hardware_concurrency();

	auto strip_height = dim / num_threads;

	vector<future<vector<double>>> futures;

	for (unsigned int i = 0; i < num_threads; i++)
		futures.push_back(async(mandelbrot, i * strip_height, (i + 1) * strip_height));

	vector<vector<double>> results;

	for (auto &f : futures)
		results.push_back(f.get());
		

	return 0;
}

