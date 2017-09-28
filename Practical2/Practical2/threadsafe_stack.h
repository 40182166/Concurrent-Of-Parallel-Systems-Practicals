#pragma once
#include <exception>
#include <stack>
#include <memory>
#include <mutex>

using namespace std;

template<typename T>
class threadsafe_stack
{
private:
	stack<T> data;
	mutable mutex mut;		//Mutable --> can be modified in const methods
public:
	threadsafe_stack() {};
	//Copy constructor --> locks other stack to ensure correct copy
	threadsafe_stack(const threadsafe_stack &other) 
	{
		lock_guard<mutex> lock(other.mut);
		data = other.data;
	};

	~threadsafe_stack() {};
	void push(T value);
	T pop();
	bool empty() const;
};

template<typename T>
void threadsafe_stack<T>::push(T value)
{
	lock_guard<mutex> lock(mut);
	data.push(value);
}

template<typename T>
T threadsafe_stack<T>::pop()
{
	lock_guard<mutex> lock(mut);
	if (data.empty())
	{
		throw exception("Stack is empty! Can't pop anything!!");
	}

	//value at top of stack
	auto res = data.top();

	//remove top item
	data.pop();

	return res;
}

template<typename T>
bool threadsafe_stack<T>::empty() const
{
	lock_guard<mutex> lock(mut);

	return data.empty();
}

