#include "stdafx.h"
#include "guarded1.h"

void guarded1::increment()
{
	lock_guard<mutex> lock(mut);
	int x = value;
	x = x + 1;
	value = x;
}
