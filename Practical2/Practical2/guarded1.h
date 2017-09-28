#pragma once
#include <mutex>
using namespace std;

class guarded1
{
public:
	guarded1() : value(0) {}
	~guarded1() {}
	int get_value() const { return value; }
	void increment();
private:
	mutex mut;
	int value;
};

