//Single-qubit gate operation can be simulated as many 2x2 matrix multiplications


#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>
using namespace std;

__global__ void
squbitsim(const float *input_A, float *output, const float *input_B, int nq, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
   int set;
       if(i < numElements)
       {
	       int index1 = i | (int)powf(2, nq);    //////////__funnelshift_r(1, 0, nq);
       	int index2 = i & ~(int)powf(2, nq);
	set=((i & (int)powf(2, nq))> 0)?1:0;
	if(set)
		output[i] = input_B[2] * input_A[index2] + input_B[3] * input_A[i];
	else
		output[i] = input_B[0] * input_A[i] + input_B[1] * input_A[index1];
       }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	vector<float> value;
	int numElements, nq;
	float *input_B= (float *)malloc(sizeof(float));
	fstream f_in("input.txt", ios::in);
	if (f_in.is_open()) {
	
	int k = 0;
	while( k < 4)
	{
		f_in >>input_B[k];
		k++;
	}

	for (float f; f_in >> f;numElements++) {
   	value.insert(value.end(), f);
    	}	
	f_in.close();
	}
	else {
    		// handle error opening file
	}
	
	nq = (int)value.back();
	value.pop_back();
	numElements--;
	size_t size = numElements *sizeof(float);
	float *input_A= (float *)malloc(size);
	float *output= (float *)malloc(size);
	
		
	// Verify that allocations succeeded
    if (input_A == NULL || output == NULL || input_B == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
	
    //Populate input array
    for(int i=0; i<numElements; ++i) 
	//std::vector<float> input_A = value;
	    input_A[i]=value[i];

    // Allocate the device input vector in
    float *d_input_A = NULL;
    err = cudaMalloc((void **)&d_input_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input_A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector output
    float *d_output = NULL;
    err = cudaMalloc((void **)&d_output, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_input_B = NULL;
    err = cudaMalloc((void **)&d_input_B, size) ; 

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input d_input_A and d_input_B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input_A, input_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input array d_input_A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input_B, input_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy qbit gate d_input_B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the squbitsim CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    squbitsim<<<blocksPerGrid, threadsPerBlock>>>(d_input_A, d_output, d_input_B, nq, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch squbitsim kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output array d_output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        printf(" %0.3f \n", output[i]);
    }
	
    //printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_input_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_input_A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_input_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_u (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(input_A);
    free(output);
    free(input_B);

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}


