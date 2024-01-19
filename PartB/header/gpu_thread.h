

#include <cuda_runtime.h>

// Create other necessary functions here

// Fill in this function

__global__ void convolutionKernel(int input_row, int input_col, int* input,
                                  int kernel_row, int kernel_col, int* kernel,
                                  int output_row, int output_col,
                                  long long unsigned int* output) {

    

    int output_i = blockIdx.y * blockDim.y + threadIdx.y;
    int output_j = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_i < output_row && output_j < output_col) {
        for (int kernel_i = 0; kernel_i < kernel_row; ++kernel_i) {
            for (int kernel_j = 0; kernel_j < kernel_col; ++kernel_j) {
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int input_j = (output_j + 2 * kernel_j) % input_col;
                output[output_i * output_col + output_j] +=
                    input[input_i * input_col + input_j] * kernel[kernel_i * kernel_col + kernel_j];
            }
        }
    }

}
             

void gpuThread( int input_row, int input_col, int* input,
                    int kernel_row, int kernel_col, int* kernel,
                    int output_row, int output_col,
                    long long unsigned int* output) 
{   
    int inputSize = input_row * input_col * sizeof(int);
    int kernelSize = kernel_row * kernel_col * sizeof(int);
    int outputSize = output_row * output_col * sizeof(long long unsigned int);

    int* d_input, *d_kernel, *d_output;

    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_kernel, kernelSize);
    cudaMalloc((void**)&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32); // Adjust the block size as needed
    dim3 numBlocks((output_col + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_row + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolutionKernel<<<numBlocks, threadsPerBlock>>>(input_row, input_col, d_input,
                                                      kernel_row, kernel_col, d_kernel,
                                                      output_row, output_col, (long long unsigned int *)d_output);

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

