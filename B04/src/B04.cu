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



/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */



__global__ void vecSum_GPU3(const double* in, double* res, const unsigned long n)
{

	//dynamic shared memory size
	__shared__ double tmp[1024];

	unsigned long i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i<n)
		tmp[threadIdx.x] = in[i];

	__syncthreads();

	//do reduction in shared memory

	for(unsigned int s=blockDim.x/2; s>0; s>>=1)
	{
		if(threadIdx.x < s)
		{
			tmp[threadIdx.x] += tmp[threadIdx.x + s];
		}
		__syncthreads();
	}


	if(threadIdx.x == 0) res[blockIdx.x] = tmp[0];
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {

	if(argc<2)
	{
		printf("Not enough Arguments, please specify a size, Nigga!\n");
		return 1;
	}

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

	long vec_size = atol(argv[1]);

	printf("Size of Vector: %d\n", vec_size);

	int blocks = ceil((float)vec_size/THREADS_PER_BLOCK);

	printf("Blocks: %d\n",blocks);

	long vec_size_full = THREADS_PER_BLOCK * blocks;					// Vector with Threads Per Block

	printf("Size of Block Filling Vector: %d\n", vec_size);

	double* vec = (double*)malloc(sizeof(double) * vec_size_full);
	double* res = (double*)malloc(sizeof(double) * vec_size_full);



	for(int i = 0; i < vec_size_full; i++)
	{
		if (i < vec_size)
			vec[i]=1.0f;
		else
			vec[i]=0.0f;

		res[i]=0;
	}


	printf("\n");

	double* d_vec;
	double* d_res;

	cudaMalloc((double **) &d_vec, vec_size_full * sizeof(double));
	cudaMalloc((double **) &d_res, vec_size_full * sizeof(double));

	cudaMemcpy(d_vec,vec,vec_size_full*sizeof(double), cudaMemcpyHostToDevice);


	vecSum_GPU3<<<blocks,THREADS_PER_BLOCK>>>(d_vec,d_res,vec_size);


	cudaMemcpy(res, d_res, vec_size_full*sizeof(double), cudaMemcpyDeviceToHost);


	for(int i = 0; i < blocks; i++)
	{
			printf("%f\n", res[i]);
	}

	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
