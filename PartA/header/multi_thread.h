

#include <pthread.h>
#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#define NUM_THREADS 8

struct ThreadData {
    int input_row, input_col, kernel_row, kernel_col, output_row, output_col;
    int *input, *kernel;
    long long unsigned int *output;
    int start_output_i, end_output_i;
};

void *threadedConvolution(void *arg) {
    struct ThreadData *data = (struct ThreadData *)arg;
    register int output_index;
    register int kernel_index;
    register int input_index;
    register int kernel_j3;
    register int kernel_j2;
    register int kernel_j1;
            	
    register int kernel_j3_2;
    register int kernel_j2_2;
    register int kernel_j1_2;
    register int kernel_j_2;
            	
    register int kernel_index_3;
    register int kernel_index_2;
    register int kernel_index_1;
    register int kernel_index_0;
    
    register int k3;
    register int k2;
    register int k1;
    register int k0;
    
    register int op1;
    register int op2;
    
    //for(int i = 0; i<data->output_row * data->output_col; i++)
    	//data->output[i] = 0;
    
    for (register int output_i = data->start_output_i; output_i < data->end_output_i; ++output_i) {
        output_index = output_i * data->output_col;

        for (register int kernel_i = 0; kernel_i < data->kernel_row; ++kernel_i) {
            kernel_index = kernel_i * data->kernel_col;
            register int input_i = (output_i + (kernel_i << 1)) % data->input_row;
            input_index = input_i * data->input_col;
	    
            for (register int kernel_j = 0; kernel_j < data->kernel_col; kernel_j += 4) {
            	kernel_j3 = (kernel_j + 3);
            	kernel_j2 = (kernel_j + 2);
            	kernel_j1 = (kernel_j + 1);
            	
            	kernel_j3_2 = kernel_j3 << 1;
            	kernel_j2_2 = kernel_j2 << 1;
            	kernel_j1_2 = kernel_j1 << 1;
            	kernel_j_2 = kernel_j << 1;
            	
            	kernel_index_3 = kernel_index + kernel_j3;
            	kernel_index_2 = kernel_index + kernel_j2;
            	kernel_index_1 = kernel_index + kernel_j1;
            	kernel_index_0 = kernel_index + kernel_j;
            	
                for (register int output_j = 0; output_j < data->output_col; output_j += 2) {
                    register int output_j1 = output_j + 1;

                    register int input_j = (output_j + (kernel_j << 1)) % data->input_col;
                    __m256i input_vec;
                    if (input_j + 7 >= data->input_col) {
                        input_vec = _mm256_set_epi32(
                            data->input[input_index + ((output_j1) + (kernel_j3_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j3_2)) % data->input_col],
                            data->input[input_index + ((output_j1) + (kernel_j2_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j2_2)) % data->input_col],
                            data->input[input_index + ((output_j1) + (kernel_j1_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j1_2)) % data->input_col],
                            data->input[input_index + ((output_j1) + (kernel_j_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j_2)) % data->input_col]
                        );
                    } else {
                        input_vec = _mm256_loadu_si256((__m256i_u *)&data->input[input_index + input_j]);
                    }

                    k3 = (kernel_j3 < data->kernel_col) ? data->kernel[kernel_index_3] : 0;
                    k2 = (kernel_j2 < data->kernel_col) ? data->kernel[kernel_index_2] : 0;
                    k1 = (kernel_j1 < data->kernel_col) ? data->kernel[kernel_index_1] : 0;
                    k0 = (kernel_j < data->kernel_col) ? data->kernel[kernel_index_0] : 0;

                    __m256i kernel_vec = _mm256_set_epi32(k3, k3, k2, k2, k1, k1, k0, k0);
                    __m256i output_vec = _mm256_mullo_epi32(input_vec, kernel_vec);

                    output_vec = _mm256_shuffle_epi32(output_vec, _MM_SHUFFLE(3, 1, 2, 0));

                    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(output_vec),
                                                    _mm256_extracti128_si256(output_vec, 1));

                    op1 = _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);
                    op2 = _mm_extract_epi32(sum128, 2) + _mm_extract_epi32(sum128, 3);

                    data->output[output_index + output_j] += op1;
                    if (output_j1 < data->output_col)
                        data->output[output_index + output_j1] += op2;
                }
            }
        }
    }

    pthread_exit(NULL);
}

void multiThread(int input_row, int input_col, int *input, int kernel_row, int kernel_col,
                 int *kernel, int output_row, int output_col, long long unsigned int *output) {

    pthread_t threads[NUM_THREADS];
    struct ThreadData thread_data[NUM_THREADS];

    register int rows_per_thread = output_row / NUM_THREADS;
    register int remaining_rows = output_row % NUM_THREADS;

    register int start_row = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_data[i].input_row = input_row;
        thread_data[i].input_col = input_col;
        thread_data[i].kernel_row = kernel_row;
        thread_data[i].kernel_col = kernel_col;
        thread_data[i].output_row = output_row;
        thread_data[i].output_col = output_col;
        thread_data[i].input = input;
        thread_data[i].kernel = kernel;
        thread_data[i].output = output;

        thread_data[i].start_output_i = start_row;
        thread_data[i].end_output_i = start_row + rows_per_thread;
        if (i == NUM_THREADS - 1) {
            thread_data[i].end_output_i += remaining_rows; // Add remaining rows to the last thread
        }

        pthread_create(&threads[i], NULL, threadedConvolution, (void *)&thread_data[i]);

        start_row = thread_data[i].end_output_i;
    }

    for (register int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
}



//COlUMN Wise distribution of data to threads
/*

#pragma GCC diagnostic ignored "-Wregister"
#include <pthread.h>
#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#define NUM_THREADS 10

struct ThreadData {
    int input_row, input_col, kernel_row, kernel_col, output_row, output_col;
    int *input, *kernel;
    long long unsigned int *output;
    int start_output_i, end_output_i;
};

void *threadedConvolution(void *arg) {
    struct ThreadData *data = (struct ThreadData *)arg;
    register int output_index;
    register int kernel_index;
    register int input_index;
    register int kernel_j3;
    register int kernel_j2;
    register int kernel_j1;
            	
    register int kernel_j3_2;
    register int kernel_j2_2;
    register int kernel_j1_2;
    register int kernel_j_2;
            	
    register int kernel_index_3;
    register int kernel_index_2;
    register int kernel_index_1;
    register int kernel_index_0;
    
    register int k3;
    register int k2;
    register int k1;
    register int k0;
    
    register int op1;
    register int op2;
    
    for (register int output_i = data->start_output_i; output_i < data->end_output_i; ++output_i) {
        output_index = output_i * data->output_col;

        for (register int kernel_i = 0; kernel_i < data->kernel_row; ++kernel_i) {
            kernel_index = kernel_i * data->kernel_col;
            register int input_i = (output_i + (kernel_i << 1)) % data->input_row;
            input_index = input_i * data->input_col;
	    
            for (register int kernel_j = 0; kernel_j < data->kernel_col; kernel_j += 4) {
            	kernel_j3 = (kernel_j + 3);
            	kernel_j2 = (kernel_j + 2);
            	kernel_j1 = (kernel_j + 1);
            	
            	kernel_j3_2 = kernel_j3 << 1;
            	kernel_j2_2 = kernel_j2 << 1;
            	kernel_j1_2 = kernel_j1 << 1;
            	kernel_j_2 = kernel_j << 1;
            	
            	kernel_index_3 = kernel_index + kernel_j3;
            	kernel_index_2 = kernel_index + kernel_j2;
            	kernel_index_1 = kernel_index + kernel_j1;
            	kernel_index_0 = kernel_index + kernel_j;
            	
                for (register int output_j = 0; output_j < data->output_col; output_j += 2) {
                    register int output_j1 = output_j + 1;

                    register int input_j = (output_j + (kernel_j << 1)) % data->input_col;
                    __m256i input_vec;
                    if (input_j + 7 >= data->input_col) {
                        input_vec = _mm256_set_epi32(
                            data->input[input_index + ((output_j1) + (kernel_j3_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j3_2)) % data->input_col],
                            data->input[input_index + ((output_j1) + (kernel_j2_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j2_2)) % data->input_col],
                            data->input[input_index + ((output_j1) + (kernel_j1_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j1_2)) % data->input_col],
                            data->input[input_index + ((output_j1) + (kernel_j_2)) % data->input_col],
                            data->input[input_index + (output_j + (kernel_j_2)) % data->input_col]
                        );
                    } else {
                        input_vec = _mm256_loadu_si256((__m256i_u *)&data->input[input_index + input_j]);
                    }

                    k3 = (kernel_j3 < data->kernel_col) ? data->kernel[kernel_index_3] : 0;
                    k2 = (kernel_j2 < data->kernel_col) ? data->kernel[kernel_index_2] : 0;
                    k1 = (kernel_j1 < data->kernel_col) ? data->kernel[kernel_index_1] : 0;
                    k0 = (kernel_j < data->kernel_col) ? data->kernel[kernel_index_0] : 0;

                    __m256i kernel_vec = _mm256_set_epi32(k3, k3, k2, k2, k1, k1, k0, k0);
                    __m256i output_vec = _mm256_mullo_epi32(input_vec, kernel_vec);

                    output_vec = _mm256_shuffle_epi32(output_vec, _MM_SHUFFLE(3, 1, 2, 0));

                    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(output_vec),
                                                    _mm256_extracti128_si256(output_vec, 1));

                    op1 = _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);
                    op2 = _mm_extract_epi32(sum128, 2) + _mm_extract_epi32(sum128, 3);

                    data->output[output_index + output_j] += op1;
                    if (output_j1 < data->output_col)
                        data->output[output_index + output_j1] += op2;
                }
            }
        }
    }

    pthread_exit(NULL);
}

void multiThread(int input_row, int input_col, int *input, int kernel_row, int kernel_col,
                 int *kernel, int output_row, int output_col, long long unsigned int *output) {

    pthread_t threads[NUM_THREADS];
    struct ThreadData thread_data[NUM_THREADS];

    // Calculate columns per thread and remaining columns
    int cols_per_thread = output_col / NUM_THREADS;
    int remaining_cols = output_col % NUM_THREADS;

    int start_col = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_data[i].input_row = input_row;
        thread_data[i].input_col = input_col;
        thread_data[i].kernel_row = kernel_row;
        thread_data[i].kernel_col = kernel_col;
        thread_data[i].output_row = output_row;
        thread_data[i].output_col = output_col;
        thread_data[i].input = input;
        thread_data[i].kernel = kernel;
        thread_data[i].output = output;

        thread_data[i].start_output_i = start_col;
        thread_data[i].end_output_i = start_col + cols_per_thread;
        if (i == NUM_THREADS - 1) {
            thread_data[i].end_output_i += remaining_cols; // Add remaining columns to the last thread
        }

        pthread_create(&threads[i], NULL, threadedConvolution, (void *)&thread_data[i]);

        start_col = thread_data[i].end_output_i;
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
}

*/


