/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}



__global__ void add(int *a, int *b, int *c)
{

	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */

#define N 1024

int main(void)
{


	int *a, *b, *c; 				// host copies of a, b, c
	int *d_a, *d_b, *d_c; 			// device copies of a, b, c
	int size = N * sizeof(int);


	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Setup input values

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	int i = 0;
	for(i = 0; i < N; i++)
	{
		a[i] = i;//rand() % 100;
		b[i] = i;//rand() % 100;
	}

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N blocks
	add<<<N,1>>>(d_a, d_b, d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(i = 0; i < N; i++)
	{
		printf("%d\n",c[i]);
	}


	// Cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}