#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

#define BLOCK_DIM 1024

__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N)
{
	unsigned int i= blockIdx.x*blockDim.x+threadIdx.x;
	output[i]=input[i];
	__syncthreads();

	for(int stride = 1;stride <=BLOCK_DIM/2; stride *=2)
	{
		float v;
		if(threadIdx.x >=stride) 
		{// include some checks for boundary conditions or out-of-bounds 
		// condition
			v = output[i-stride]; 
		}
		__syncthreads();
		if(threadIdx.x >=stride)
		{ 
			output[i] +=v;
		}
		__syncthreads();
	}
	if(threadIdx.x == BLOCK_DIM-1)
	{
		partialSums[blockIdx.x] = output[i];
	}
}

__global__ void scan_kernel_shmem(float* input, float* output, float* partialSums, unsigned int N)
{
	unsigned int i= blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ float buffer_s[BLOCK_DIM];

	buffer_s[threadIdx.x]=input[i];
	__syncthreads();

	for(int stride = 1;stride <=BLOCK_DIM/2; stride *=2)
	{
		float v;
		// why two syncthreads?
		// to make sure everybody reads before everybody writes
		if(threadIdx.x >=stride) 
		{// include some checks for boundary conditions or out-of-bounds 
		// condition and  
			v = buffer_s[threadIdx.x-stride]; 
		}
		__syncthreads();
		if(threadIdx.x >=stride)
		{ 
			buffer_s[threadIdx.x] +=v;
		}
		__syncthreads();
	}
	if(threadIdx.x == BLOCK_DIM-1)
	{
		partialSums[blockIdx.x] = buffer_s[threadIdx.x];
	}
	output[i] = buffer_s[threadIdx.x];
}

__global__ void scan_kernel_doubleBuffer(float* input, float* output, float* partialSums, unsigned int N)
{
	unsigned int i= blockIdx.x*blockDim.x+threadIdx.x;
	__shared__ float buffer1_s[BLOCK_DIM];
	__shared__ float buffer2_s[BLOCK_DIM]; 

	float* inBuffer_s = buffer1_s;
	float* outBuffer_s = buffer2_s; 

	inbuffer_s[threadIdx.x]=input[i];
	__syncthreads();

	for(int stride = 1;stride <=BLOCK_DIM/2; stride *=2)
	{
		if(threadIdx.x > = stride)
		{
			outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
		}
		else
		{
			outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
		}
		__syncthreads();

		float* temp = inBuffer_s;
		inBuffer_s = outBuffer_s; 
		outBuffer_s = temp; 
		
 	}

	if(threadIdx.x == BLOCK_DIM-1)
	{
		partialSums[blockIdx.x] = inBuffer_s[threadIdx.x];
	}
	output[i] = inBuffer_s[threadIdx.x];
}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(blockIdx.x >0)
		output[i] += partialSums[blockIdx.x -1];
}


void scan_gpu_d(float* input, float* output, unsigned int N)
{

	//Configs 
	const unsigned int numThreadsPerBlock = BLOCK_DIM; 
	const unsigned int numElementsPerBlock = numThreadsPerBlock;
	const unsigned int numBlocks = (N + numElementsPerBlock-1)/numElementsPerBlock;

	//Allocate partial Sums,input and output;
	float* input_d, *output_d, float* partialSums_d; 
	cudaMallod((void**) &input_d, N*sizeof(float));
	cudaMallod((void**) &output_d, N*sizeof(float));
	cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
	
	cudaMemcpy(input_d, input,N*sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	//Call Kernel
	scan_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d,N);
	cudaDeviceSynchronize();


	if(numBlocks>1)
	{
		//Scan Partial Sums - recursive calls
		scan_gpu_d(input_d, output_d,N);

		// Add scanned sums
		add_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d,partialSums_d,N);

	}
	
	//cudaMemcpy(output_d,N*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();



	//stopTimer will come
	//validate on cpu;

/*	for(int i=0;i<N;i++)
	{
		input_d[i]=input_d[i-1] + input_d[i];
		assert input_d[i] = output_d[i];
	}
*/


	cudaFree(partialSums_d);
	cudaDeviceSynchronize();

}



int main(){
	int arraySize=65536;
	float input[arraySize];
	float output[arraySize];
	srand(time(0));
	

	for(int i = 0; i < arraySize; ++i)
	{
		input[i] = 1.0f + (float)(rand() % 100);
	}

	//totalSum = reduce_gpu(a,arraySize);
	scan_gpu_d(input,output,arraySize);

	//printf("Total Sum of the array: %f", totalSum);

}