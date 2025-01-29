#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

#define BLOCK_DIM 512
#define COARSE_FACTOR 1

using namespace std;

//assuming size of array is twice the size of BLOCK_DIM;

__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N){
	unsigned int segment = blockIdx.x * blockDim.x * 2;
	unsigned int i = segment + threadIdx.x * 2;
	for (unsigned int stride = 1; stride <= BLOCK_DIM; stride*=2){
		if(threadIdx.x%stride==0){
			input[i] += input[i+stride];
		}
		__syncthreads();
	} 
	if(threadIdx.x == 0){
		partialSums[blockIdx.x] = input[i];
	}
}

__global__ void reduce_kernel_controldiv(float* input, float* partialSums, unsigned int N){
	unsigned int segment =blockIdx.x * blockDim.x *2;
	unsigned int i = segment + threadIdx.x;
	for(unsigned int stride = BLOCK_DIM; stride >0;stride /=2){
		if(threadIdx.x < stride)
		{
			input[i] +=input[i+stride];
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		partialSums[blockIdx.x] = input[i];
	}
}

__global__ void reduce_kernel_sharedmem(float* input, float* partialSums, unsigned int N){
	unsigned int segment =blockIdx.x * blockDim.x *2;
	unsigned int i = segment + threadIdx.x;
	__shared__ float input_s[BLOCK_DIM];
	input_s[threadIdx.x] = input[i] + input[i+BLOCK_DIM];
	__syncthreads();
	for(unsigned int stride = BLOCK_DIM/2; stride >0; stride /=2){
		if(threadIdx.x < stride)
		{
			input_s[threadIdx.x] +=input_s[threadIdx.x+stride];
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		partialSums[blockIdx.x] = input_s[threadIdx.x];
	}
}

__global__ void reduce_kernel_sharedmem_coarsefactor(float* input, float* partialSums, unsigned int N){
	unsigned int segment =blockIdx.x * blockDim.x *2 * COARSE_FACTOR;
	unsigned int i = segment + threadIdx.x;
	__shared__ float input_s[BLOCK_DIM];
	float sum=0.0f;
	for(unsigned int tile = 0;tile < COARSE_FACTOR*2; tile++)
	{
		sum += input[i + tile*BLOCK_DIM];
	}

	input_s[threadIdx.x]=sum;
	__syncthreads();

	for(unsigned int stride = BLOCK_DIM/2; stride >0; stride /=2){
		if(threadIdx.x < stride)
		{
			input_s[threadIdx.x] +=input_s[threadIdx.x+stride];
		}
		__syncthreads();
	}
	if(threadIdx.x==0){
		partialSums[blockIdx.x] = input_s[threadIdx.x];
	}
}




float reduce_gpu(float* input, unsigned int N){
	

	//Allocate memory 
	
	float* input_d;	
	float* partialSums_d;
	float sum;

	cudaMalloc((void**) &input_d, N*sizeof(float));
	cudaDeviceSynchronize();
	
	//Copy data to GPU 
	cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	//Allocate partial sums
	const unsigned int numThreadsPerBlock = BLOCK_DIM;
	//const unsigned int numElementsPerBlock = 2*numThreadsPerBlock*COARSE_FACTOR;
	const unsigned int numElementsPerBlock = numThreadsPerBlock/4 * COARSE_FACTOR;
	const unsigned int numBlocks = (N + numElementsPerBlock -1)/numElementsPerBlock;
	float* partialSums = (float*) malloc(numBlocks*sizeof(float));

	cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
	cudaDeviceSynchronize();
	

	//Call kernel 
//	reduce_kernel<<< numBlocks, numThreadsPerBlock >>>(input_d, partialSums_d, N);
//	cudaDeviceSynchronize();
	
	//reduce_kernel_controldiv<<< numBlocks, numThreadsPerBlock >>>(input_d, partialSums_d, N);
	//cudaDeviceSynchronize();

	//reduce_kernel_sharedmem<<< numBlocks, numThreadsPerBlock >>>(input_d, partialSums_d, N);
	//cudaDeviceSynchronize();

	reduce_kernel_sharedmem_coarsefactor<<< numBlocks, numThreadsPerBlock >>>(input_d, partialSums_d, N);
	cudaDeviceSynchronize();

	//Copy data from GPU
	cudaMemcpy(partialSums, partialSums_d, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	

	//Reduce partial Sums on the CPU
	for(unsigned int i=0; i<numBlocks; i++){
		sum+=partialSums[i];
	}

	//Free memory
	cudaFree(input_d);
	free(partialSums);
	cudaFree(partialSums_d);
	cudaDeviceSynchronize();
	

	return sum;
}

int main(){
	int arraySize=65536;
	float a[arraySize];
	srand(time(0));
	float totalSum;

	for(int i = 0; i < arraySize; ++i)
	{
		a[i] = 1.0f + (float)(rand() % 100);
	}

	totalSum = reduce_gpu(a,arraySize);

	printf("Total Sum of the array: %f", totalSum);

}