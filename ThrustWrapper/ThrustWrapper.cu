#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

extern void sort_uint_internal(void* dev_ptr, unsigned numElements, void* output_ptr)
{
	if(output_ptr) {
		cudaMemcpy(output_ptr, dev_ptr, numElements * sizeof(unsigned), cudaMemcpyDeviceToDevice);
	} else {
		output_ptr = dev_ptr;
	}
	thrust::device_ptr<unsigned> dp((unsigned*)output_ptr);
	thrust::stable_sort(dp, dp + numElements);
}

extern void sort_double_internal(void* dev_ptr, unsigned numElements, void* output_ptr)
{
	if(output_ptr) {
		cudaMemcpy(output_ptr, dev_ptr, numElements * sizeof(double), cudaMemcpyDeviceToDevice);
	} else {
		output_ptr = dev_ptr;
	}
	thrust::device_ptr<double> dp((double*)output_ptr);
	thrust::stable_sort(dp, dp + numElements);
}