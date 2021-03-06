#define _CRT_RAND_S

#include <time.h>
#include <winsock.h>
#include <stdint.h>
#include <stdlib.h>  
#include <stdio.h>  
#include <limits.h> 

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

void timersub(struct timeval *a, struct timeval *b, struct timeval *res)
{
	res->tv_sec = a->tv_sec - b->tv_sec;
	if (a->tv_usec - b->tv_usec >= 0)
	{
		res->tv_usec = a->tv_usec - b->tv_usec;
	}
	else{
		res->tv_sec--;
		res->tv_usec = a->tv_usec + 1000000 - b->tv_usec;
	}
}


extern DWORD main_thread;
int rand_r(void)
{
	int ret;
#ifndef LIBDARKNET
	if (main_thread == GetCurrentThreadId())
	{
		ret = rand();
	}
	else
	{
		unsigned int index;
		rand_s(&index);
		ret = (unsigned int)((double)index / ((double)UINT_MAX + 1) * RAND_MAX);
	}
#else
	ret = rand();
#endif
	return ret;
}
