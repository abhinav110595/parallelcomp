
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iterator>
#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
using namespace std;

__global__ void
quamsim(float *input_A, float *input_B, float *output,int nq,  int numElements, int index_incr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int gate_index = index_incr * 4;
    int index1 = i | (int)powf(2, nq);
        if(i < numElements ) 
        {
		int set = i % int (1<<(nq+1))<int (1<<(nq));
		if(set)
            output[i] = (input_B[gate_index + 0] * input_A[i] + input_B[gate_index + 1] * input_A[index1]);
            output[index1] = (input_B[gate_index + 2] * input_A[i] + input_B[gate_index + 3] * input_A[index1]);

        }
    
}

/**
 * Host main routine
 */


int main(int argc, char* argv[])
{

    FILE * myfile;
   // myfile=fopen(input_file,"r");
    myfile=fopen(argv[1],"r");
    int linesCount=0;
    float* input_B=new float [24];


    float nq[6];
    if(myfile==NULL){
        cout<<"File not found"<<endl;
        return 0;
    }


ifstream MyReadFile(argv[1]);
   string  myText;
    while (getline (MyReadFile, myText)) {
     linesCount=linesCount+1;

    }
    float* input_A = new float [(linesCount-25)];
    float* output = new float [(linesCount-25)];
/*
    	std::ifstream infile(argv[1]);
	if (!infile.is_open()) {
    	std::cout << "Error opening file" << std::endl;
    	return 0;

  */ 
      myfile=fopen(argv[1],"r");
    int i=0;
    while(fscanf(myfile, "%f %f", &input_B[i], &input_B[i+1]) != EOF )
    {
        i+=2;
        if (i>23)
        {
            i = 0;
            while (fscanf(myfile, "%f ", &input_A[i]) != EOF)
            {
            if(i>linesCount-27)
                    break;
               i++;
            }
            i=0;
            while (fscanf(myfile, "%f", &nq[i]) != EOF)
            {
                if (i>(4))
            break;
                i++;
        }
            break;
    }
    }

    cudaError_t err = cudaSuccess;

    int numElements = linesCount-25;
    size_t size = numElements  * sizeof(float);
    size_t size_B = 24 * sizeof(float);
    size_t size_out = numElements * sizeof(float);



    if (input_A == NULL || input_B == NULL || output == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    float *d_input_A = NULL;
    err = cudaMalloc((void **)&d_input_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input_A  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Allocate the device input vector B
    float *d_input_B = NULL;
    err = cudaMalloc((void **)&d_input_B, size_B);

    float *d_output1 = NULL;
    err = cudaMalloc((void **)&d_output1, size_out);
    
    float *d_output2 = NULL;
    err = cudaMalloc((void **)&d_output2, size_out);
    
    float *d_output3 = NULL;
    err = cudaMalloc((void **)&d_output3, size_out);
    
    float *d_output4 = NULL;
    err = cudaMalloc((void **)&d_output4, size_out);
    
    float *d_output5 = NULL;
    err = cudaMalloc((void **)&d_output5, size_out);
    

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input_B  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_output = NULL;
    err = cudaMalloc((void **)&d_output, size_out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input_A, input_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input_A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_input_B, input_B, size_B, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input_B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 // Launch the  Kernel

    	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	float* input_A_ptr = d_input_A;
	float* output_ptr = d_output1;

	for(int i=0; i<6; i++){
    	quamsim<<<blocksPerGrid, threadsPerBlock>>>(input_A_ptr, d_input_B, output_ptr, int(nq[i]), numElements, i);
    	input_A_ptr = output_ptr;
    	output_ptr = i == 4 ? d_output : output_ptr + numElements;
}

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(output, d_output, size_out, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for(i=0;i<numElements;i++)
    {
        cout<<fixed<<setprecision(3)<<output[i]<<endl;
    }
    // Verify that the result vector is correct
    // Free device global memory
    err = cudaFree(d_input_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_input_A  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_input_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_input_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_output  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free host memory
    free(input_A);
    free(input_B);
    free(output);


    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

