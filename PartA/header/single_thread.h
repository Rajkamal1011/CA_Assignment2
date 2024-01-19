
#pragma GCC diagnostic ignored "-Wregister"
#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

// Optimize this function
void singleThread(register int input_row,register int input_col,register int *input,register int kernel_row,
                  register int kernel_col,register int *kernel,register int output_row,register int output_col,
                  register long long unsigned int *output) {

  register int input_index;
  register int kernel_index;
  register int output_index;
  
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
    
    for(int i = 0; i < output_row * output_col; ++i)
        output[i] = 0;
    

    
  for (register int output_i = 0; output_i < output_row; ++output_i) { // A
  		output_index = output_i * output_col;
  	for (register int kernel_i = 0; kernel_i < kernel_row; ++kernel_i) {   // C
  		kernel_index = kernel_i * kernel_col;
  	
  	
        	register int input_i = (output_i + (kernel_i << 1)) % input_row;
        	input_index = input_i * input_col;
        	
        	
      		for (register int kernel_j = 0; kernel_j < kernel_col; kernel_j += 4) {   // D
      			//Following precomputation helps in non-repeatation of computation
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
      			
        		for (register int output_j = 0; output_j < output_col; output_j += 2) { // B
        			register int output_j1 = output_j + 1;
          		
          			register int input_j = (output_j + (kernel_j_2)) % input_col;
          			__m256i input_vec;
          			if (input_j + 7 >= input_col) {
            				input_vec = _mm256_set_epi32(
                				input[input_index + ((output_j1) + (kernel_j3_2)) % input_col],
                			
                				input[input_index + (output_j + (kernel_j3_2)) % input_col],
                				
                				input[input_index + ((output_j1) + (kernel_j2_2)) % input_col],
                				
            					input[input_index + (output_j + (kernel_j2_2)) % input_col],
            					
                				input[input_index + ((output_j1) + (kernel_j1_2)) % input_col],
                				
                				input[input_index + (output_j + (kernel_j1_2)) % input_col],
                			
                				input[input_index + ((output_j1) + (kernel_j_2)) % input_col],
                			
                				input[input_index + (output_j + (kernel_j_2)) % input_col]);

          			} else {
            				input_vec = _mm256_loadu_si256(
                			(const __m256i_u *)&input[input_index + input_j]);
          			}

          			k3 = (kernel_j3 < kernel_col) ? kernel[kernel_index_3]: 0;
          		
          			k2 = (kernel_j2 < kernel_col) ? kernel[kernel_index_2]: 0;
          		
          			k1 = (kernel_j1 < kernel_col)? kernel[kernel_index_1]: 0;
          		
          			k0 = (kernel_j < kernel_col)? kernel[kernel_index_0]: 0;

          			__m256i kernel_vec = _mm256_set_epi32(k3, k3, k2, k2, k1, k1, k0, k0);
          			__m256i output_vec = _mm256_mullo_epi32(input_vec, kernel_vec);
          		
          			output_vec = _mm256_shuffle_epi32(output_vec, _MM_SHUFFLE(3, 1, 2, 0));
          		
          			__m128i sum128 =
              				_mm_add_epi32(_mm256_castsi256_si128(output_vec),
                            			_mm256_extracti128_si256(output_vec, 1));
                            
          			auto out1 = _mm_extract_epi32(sum128, 0) + _mm_extract_epi32(sum128, 1);
          			auto out2 = _mm_extract_epi32(sum128, 2) + _mm_extract_epi32(sum128, 3);
          
          			output[output_index + output_j] += out1;
          			if(output_j1 < output_col)
          				output[output_index + output_j1] += out2;
        		}
      		}
    	}
  }
}



