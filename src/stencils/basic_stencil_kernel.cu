#include "common.h"
#include "timer.h"

#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM-2)


__global__ void stencil_kernel(float* in, float* out, unsigned int N){
// Not doing boundary conditions.. so no indices=0 or N-1
	
	unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
	if((i>=1 && i< N-1) && (j>=1 && j< N-1) && (k>=1 && k< N-1))
	{
		out[i*N*N+j*N+k]= c0*in[i*N*N + j*N + k] +
						  c1*(in[i*N*N + j*N + (k-1)] + 
						      in[i*N*N + j*N + (k+1)] +
						      in[i*N*N + (j-1)*N + k] +
						      in[i*N*N + (j+1)*N + k]+
						      in[(i-1)*N*N + j*N + k] +
						      in[(i+1)*N*N + j*N + k]
						      );
	}
}

__global__ void stencil_smem_kernel(float* in, float* out, unsigned int N){
// Not doing boundary conditions.. so no indices=0 or N-1
	__shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

	if(i>=0 && i< N && j>=0 && j< N && k>=0 && k< N){
		in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[ i*N*N + j*N + k ];

	}

	__syncthreads();

	int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z-1;
	int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y-1;
	 int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x-1;
	if((i>=1 && i< N-1) && (j>=1 && j< N-1) && (k>=1 && k< N-1))
	{
		if(threadIdx.x >=1 && threadIdx.x < blockDim.x -1 && threadIdx.y >=1 && threadIdx.y < blockDim.y-1 && threadIdx.z>=1 && threadIdx.z < blockDim.z-1)
		out[i*N*N + j*N + k]= c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
						      c1*(in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]+
						      in[threadIdx.z][threadIdx.y][threadIdx.x+1] +
						      in[threadIdx.z][threadIdx.y-1][threadIdx.x] +
						      in[threadIdx.z][threadIdx.y+1][threadIdx.x]+
						      in[threadIdx.z-1][threadIdx.y][threadIdx.x] +
						      in[threadIdx.z+1][threadIdx.y][threadIdx.x]
						      );
	}
}

void stencil_gpu(float*in, float* out, insigned int N){
	Timer timer;

	startTime(&timer);
	float *in_d, *out_d;

	cudaMalloc((void**) &in_d, N*N*N*sizeof(float));
	cudaMalloc((void**) &out_d, N*N*N*sizeof(float));
	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Allocation time");

	//Copy data to GPU
	startTime(&timer);
	cudaMemcpy(in_d, in, N*N*N*sizeof(float), cudaMemcpyHostToDevice); 
	cudaDeviceSynchronize();
	stopTime(&timer);
	printElapsedTime(timer, "Copy to GPU time");

	// Call kernel
	startTime(&timer);
	dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 numBlocks((N + BLOCK_DIM -1)/BLOCK_DIM, (N + BLOCK_DIM -1)/BLOCK_DIM, (N + BLOCK_DIM -1)/BLOCK_DIM);
	stencil_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);

	//Call shared mem tile kernel 
	startTime(&timer);
	dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 numBlocks((N + OUT_TILE_DIM-1)/OUT_TILE_DIM, (N + OUT_TILE_DIM -1)/OUT_TILE_DIM, (N + OUT_TILE_DIM, -1)/OUT_TILE_DIM);
	stencil_smem_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);

	}

