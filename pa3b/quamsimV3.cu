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

__global__ void quamsim (float* input_A, float* input_B, float* output, int* nq, int numElements) {
    __shared__ float shared_Memory[265];
    	unsigned int MemIndex, mappedBlockId, threadId;
    	mappedBlockId = blockIdx.x;
    	threadId = threadIdx.x;
    	mappedBlockId = ((((mappedBlockId >> nq[0]) << (1)) | ((threadId) & (1))) << (nq[0])) | ((mappedBlockId) & (~((~0u) << nq[0])));
	for (int i = 1; i < 5; i++) {
		mappedBlockId = ((((mappedBlockId >> nq[i]) << (1)) | ((threadId >> i) & (1))) << (nq[i])) | ((mappedBlockId) & (~((~0u) << nq[i])));
	}
	MemIndex = ((((mappedBlockId >> nq[5]) << (1)) | ((threadId >> 5) & (1))) << (nq[5])) | ((mappedBlockId) & (~((~0u) << nq[5])));
    	shared_Memory[threadId] = input_A[MemIndex];
   for (int i = 0; i < 6; i++) {
       if ((threadId & (1 << i)) == 0) {
           shared_Memory[threadId] = input_B[4 * i] * shared_Memory[threadId] + input_B[4 * i + 1] * shared_Memory[threadId | (1 << i)];
       } else {
           shared_Memory[threadId] = input_B[4 * i + 2] * shared_Memory[threadId & (~(1 << i))] + input_B[4 * i + 3] * shared_Memory[threadId];
       }
       __syncthreads();
   }
   output[MemIndex] = shared_Memory[threadId];
}


/*
__global__ void quamsim (float* input_A, float* input_B, float* output, int* nq, int numElements) {
    __shared__ float shared_Memory[64];

    unsigned int MemIndex, mappedBlockId, threadId;
    mappedBlockId = blockIdx.x;
    threadId = threadIdx.x;
    mappedBlockId = ((((mappedBlockId >> nq[0]) << (1)) | ((threadId) & (1))) << (nq[0])) | ((mappedBlockId) & (~((~0u) << nq[0])));

    for (int i = 1; i < 5; i++) {
        mappedBlockId = ((((mappedBlockId >> nq[i]) << (1)) | ((threadId >> i) & (1))) << (nq[i])) | ((mappedBlockId) & (~((~0u) << nq[i])));
    }

    MemIndex = ((((mappedBlockId >> nq[5]) << (1)) | ((threadId >> 5) & (1))) << (nq[5])) | ((mappedBlockId) & (~((~0u) << nq[5])));
    shared_Memory[threadId] = input_A[MemIndex];

    for (int i = 0; i < 6; i += 2) {
        int j = i + 1;
        if ((threadId & (3 << i)) == 0) {
            shared_Memory[threadId] = input_B[4 * i] * shared_Memory[threadId] + input_B[4 * i + 1] * shared_Memory[threadId | (1 << i)];
            shared_Memory[threadId] = input_B[4 * j] * shared_Memory[threadId] + input_B[4 * j + 1] * shared_Memory[threadId | (1 << j)];
        }
        else if ((threadId & (3 << i)) == (1 << i)) {
            shared_Memory[threadId] = input_B[4 * i + 2] * shared_Memory[threadId & (~(1 << i))] + input_B[4 * i + 3] * shared_Memory[threadId];
            shared_Memory[threadId] = input_B[4 * j] * shared_Memory[threadId] + input_B[4 * j + 1] * shared_Memory[threadId | (1 << j)];
        }
        else if ((threadId & (3 << i)) == (2 << i)) {
            shared_Memory[threadId] = input_B[4 * i] * shared_Memory[threadId] + input_B[4 * i + 1] * shared_Memory[threadId | (1 << i)];
            shared_Memory[threadId] = input_B[4 * j + 2] * shared_Memory[threadId & (~(1 << j))] + input_B[4 * j + 3] * shared_Memory[threadId];
        }
        else {
            shared_Memory[threadId] = input_B[4 * i + 2] * shared_Memory[threadId & (~(1 << i))] + input_B[4 * i + 3] * shared_Memory[threadId];
            shared_Memory[threadId] = input_B[4 * j + 2] * shared_Memory[threadId & (~(1 << j))] + input_B[4 * j + 3] * shared_Memory[threadId];
        }
        __syncthreads();
    }
    output[MemIndex] = shared_Memory[threadId];

}

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

    myfile=fopen(argv[1],"r");

    int i=0;
    while(fscanf(myfile, "%f %f", &input_B[i], &input_B[i+1]) != EOF)
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
    size_t nq_integers_size = 6 *sizeof(int);
    int* nq_integers = new int [6];
    for(int k=0;k<6;k++)
        nq_integers[k]=int(nq[k]);
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

    int *d_nq_integers = NULL;
    err = cudaMalloc((void **)&d_nq_integers, nq_integers_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device nq_integers  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_nq_integers, nq_integers, nq_integers_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input_A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input_B  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

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
    int n = log2(float(numElements));
    int threadsPerBlock = 64;
    int blocksPerGrid =(1<<(n-6));
    quamsim<<<blocksPerGrid, threadsPerBlock>>>(d_input_A, d_input_B, d_output,d_nq_integers, numElements);

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

