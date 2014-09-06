// ThrustWrapper.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "ThrustWrapper.h"

extern void sort_uint_internal(void* dev_ptr, unsigned numElements, void* output_ptr);
extern void sort_double_internal(void* dev_ptr, unsigned numElements, void* output_ptr);

THRUSTWRAPPER_API void WINAPI GPUSortThrustUint(void* dev_ptr, unsigned numElements, OPTIONAL void* output_ptr)
{
	sort_uint_internal(dev_ptr, numElements, output_ptr);
}


THRUSTWRAPPER_API void WINAPI GPUSortThrustDouble(void* dev_ptr, unsigned numElements, OPTIONAL void* output_ptr)
{
	sort_double_internal(dev_ptr, numElements, output_ptr);
}