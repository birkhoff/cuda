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
#include <sys/time.h>
#include <math.h>

//static const int WORK_SIZE = 256;

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


__global__ void multiply(int *a, int *b, int *c, int a_r,int a_c, int b_r, int b_c)
{
	//c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	// c is a_r b_c
	int row = index/b_c;
	int column = index % b_c;

	int i;
	for(i=0; i<a_c;i++)
	{
		c[index] += a[row+i]*b[column+(i*b_c)];
	}

}


void printMatrix(int* m, int rows, int columns)
{
	int i;
	int j;
	for(i=0; i<rows;i++)
	{
		for(j=0;j<columns;j++)
		{
			printf("%d\t",m[i*columns+j]);
		}
		printf("\n");
	}
	printf("\n");
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char *argv[]) {

	struct timeval t0;
	struct timeval t1;

	int a_r = atoi(argv[1]);				// count of rows from A
	int a_c = atoi(argv[2]);				// column from A

	int b_r = atoi(argv[3]);				// count of rows from B
	int b_c = atoi(argv[4]);				// column from B

	if(a_c != b_r)
	{
		printf("\n\tError! \n\tPlease match the size of colums of A with the size of rows of B!\n\n");
		return -1;
	}

	int *a, *b, *c; 					// host copies of a, b, c
	int *d_a, *d_b, *d_c; 				// device copies of a, b, c
	//	int size = i * sizeof(int);

	a = (int *)malloc(a_r*a_c * sizeof(int));
	b = (int *)malloc(b_r*b_c * sizeof(int));
	c = (int *)malloc(a_r*b_c* sizeof(int));

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, a_r*a_c * sizeof(int));
	cudaMalloc((void **)&d_b, b_r*b_c * sizeof(int));
	cudaMalloc((void **)&d_c, a_r*b_c * sizeof(int));

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	int THREADS_PER_BLOCK = 0;

	for (int i = 0; i < nDevices; i++)
	{
	    cudaDeviceProp prop;
	    cudaGetDeviceProperties(&prop, i);
	    printf("  Device Number: %d\n", i);
	    printf("  Device name: %s\n", prop.name);
	    printf("  Memory Clock Rate (KHz): %d\n",
	           prop.memoryClockRate);
	    printf("  Memory Bus Width (bits): %d\n",
	           prop.memoryBusWidth);
	    printf("  Peak Memory Bandwidth (GB/s): %f\n",
	           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	    printf("  Max Threads Per Block: %d\n\n", prop.maxThreadsPerBlock);
	    THREADS_PER_BLOCK = prop.maxThreadsPerBlock;
	}


	int i = 0;
	int j = 0;

	for(i =0; i < a_r; i++)
	{
		for(j=0; j < a_c; j++)
		{
			a[i*a_c+j] = 1;		//rand() % 100;
		}
	}

	for(i=0; i < b_r; i++)
	{
		for(j=0; j < b_c; j++)
		{
			b[i*b_c+j] = 1;		//rand() % 100;

		}
	}

	for(i=0; i < a_r; i++)
	{
		for(j= 0; j < b_c; j++)
		{
			c[i*b_c+j] = 0;
		}
	}


	gettimeofday(&t0,0);

	// Copy inputs to device
	cudaMemcpy(d_a, a, a_r*a_c * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, b_r*b_c * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, a_r*b_c * sizeof(int), cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N blocks
	int blocks = ceil((float)a_r*b_c/1024);								// round up
	multiply<<<blocks,1024>>>(d_a, d_b, d_c, a_r,a_c, b_r, b_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, a_r*b_c * sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday(&t1,0);
	double time_spent = (t1.tv_sec-t0.tv_sec) + (double)(t1.tv_usec-t0.tv_usec)/1000000;

	printMatrix(a, a_r, a_c);
	printMatrix(b, b_r, b_c);
	printMatrix(c, a_r, b_c);
	printf("Time Calculated: %f\n\n", time_spent);
	printf("Block Count: %d\n",blocks);
	return 0;
}
