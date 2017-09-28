// Practical2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "guarded1.h"
#include "threadsafe_stack.h"
#include <memory>
#include <thread>
#include <vector>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

mutex mut;

const unsigned int NUM_ITERATIONS = 1000000;
const unsigned int NUM_THREADS = 4;


void increment(shared_ptr<int> value)
{
	for (unsigned int i = 0; i < 1000000; i++)
	{
		lock_guard<mutex> lock(mut);
		*value = *value + 1;

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

	auto stack = make_shared<threadsafe_stack<unsigned int>>();

	thread t1(popper, stack);
	thread t2(pusher, stack);

	t1.join();
	t2.join();

	cout << "Stack empty = " << stack->empty() << endl;


    return 0;
}

