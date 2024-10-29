#include <stdio.h> 
#include <stdlib.h> 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>

#define FILTER_RADIUS 1

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define THREADS_PER_BLOCK_Z 3

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void conv2dBasic_Kernel(float* N, float* P, int r, int width, int height, int channels){
	
	int outCol = blockIdx.x*blockDim.x + threadIdx.x;
	int outRow = blockIdx.y*blockDim.y + threadIdx.y;
	int channelIndex = blockIdx.z*blockDim.z + threadIdx.z; 

	float Pvalue = 0.0f;
	int filterDiameter = 2*r+1; 

	if(outCol < width && outRow < height && channelIndex < channels)
	{
		for(int fRow=0; fRow<filterDiameter; fRow++){
			for(int fCol =0; fCol<filterDiameter; fCol++){
				int inRow = outRow -r + fRow;
				int inCol = outCol -r + fCol;
				if(inRow>=0 && inRow < height &&inCol >=0 &&inCol<width){
					float fValue = F[fCol][fRow];
					float nValue = N[(inRow*width+inCol)*channels + channelIndex];	
					Pvalue += fValue * nValue;

				}
			}
		}

		P[(outRow*width+outCol)*channels + channelIndex] = Pvalue;
		//P[outRow][outCol][channels]=Pvalue;
	}
}







void conv2dBasic(
	float* N_h,
	float* F_h,
	float* P_h, 
	int r,
	int width,
	int height,
	int channels){

	float* N_d= NULL;
	float* P_d= NULL;
	int filterDiameter = 2*r+1;
	int fSize = filterDiameter*filterDiameter*sizeof(float);
	cudaMemcpyToSymbol(F, F_h,fSize);

	int nSize = width*height*channels*sizeof(float);

	cudaMalloc((void**)&N_d, nSize);
	cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
	
	cudaMalloc((void**)&P_d, nSize);
	cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
	
    cudaMemcpy(N_d,N_h,nSize,cudaMemcpyHostToDevice);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    dim3 dimBlock{
    	THREADS_PER_BLOCK_X, 
    	THREADS_PER_BLOCK_Y,
    	THREADS_PER_BLOCK_Z
    };
    dim3 dimGrid{
    	ceil((float)width / THREADS_PER_BLOCK_X ),
        ceil((float)height / THREADS_PER_BLOCK_Y ),
        1
    };

    conv2dBasic_Kernel<<< dimGrid,dimBlock >>>(N_d, P_d, r, width, height, channels);

    cudaMemcpy(P_h, P_d, nSize, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(P_d);

    return;

	}




int main(void){
	
	int height;
	int width;
	int channels;
	int r =1;

	const char* output_filename = "puppy_convolution.png";
	float* N; 
	float* P;
	unsigned char* img_out;

	float F_h[] = {
    	0.0751, 0.1238, 0.0751,
        0.1238, 0.2042, 0.1238,
        0.0751, 0.1238, 0.0751
    };
	
	unsigned char* img_in = stbi_load("puppy.jpg", &width, &height, &channels,4);

	if (img_in==NULL){
		printf("Error in loading image \n");
		exit(1);
	}
	printf("width is: %d\n", width);
	printf("height is: %d\n", height);
    printf("channels is: %d\n", channels);

	printf("Loaded stb_image");

	N = (float*) malloc(width * height * channels * sizeof(float));

	for (int i=0; i < width*height*channels; i++)
	{
		N[i]=(float)(img_in[i])/255.0f;
	}
	
	P = (float*) malloc(width*height*channels*sizeof(float));

	conv2dBasic(N, F_h, P, r, width, height, channels); 

	img_out = (unsigned char*) malloc(width * height *channels);
	for (int i=0; i<width*height*channels;i++)
	{
		img_out[i]= (unsigned char)(P[i]*255.0f);
	}

	stbi_write_png(output_filename, width, height, channels, img_out, width * channels);

  	printf("Saved grayscale image to %s\n", output_filename);
    // Clean up
    stbi_image_free(img_in);
    free(img_out);
    free(N);
    free(P);

}	