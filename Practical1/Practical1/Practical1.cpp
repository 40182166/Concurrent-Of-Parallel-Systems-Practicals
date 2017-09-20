// Practical1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <thread>
#include <iostream>
#include <chrono>
#include <random>
#include <functional>

using namespace std;
using namespace std::chrono;
using namespace std::this_thread;

const int num_threads = 100;

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

int main()
{
	thread t([] {cout << "hello from lambda thread!" << endl; });
	t.join();
	
    return 0;
}

