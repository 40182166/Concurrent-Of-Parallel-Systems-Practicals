// Practical2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
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

void increment(shared_ptr<int> value)
{
	for (unsigned int i = 0; i < 1000000; i++)
	{
		lock_guard<mutex> lock(mut);
		*value = *value + 1;

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
	if (condition.wait_for(unique_lock<mutex>(mut), seconds(3)) == true)
	{
		cout << "Task 1 got tired waiting" << endl;
	}
	cout << "Task 1 finished" << endl;
}

int main()
{
	auto value = make_shared<int>(0);
	auto num_threads = thread::hardware_concurrency();
	vector<thread> threads;
	for (unsigned int i = 0; i < num_threads; i++)
	{
		threads.push_back(thread(increment, value));
	}

	for (auto &t : threads)
	{
		t.join();
	}

	cout << "Value = " << *value << endl;

    return 0;
}

