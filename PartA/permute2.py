import os
import csv
from itertools import permutations

TIMES = 5
IN = 4096
KERN = 13

source_preamble = """
// Optimize this function
void singleThread(int input_row, int input_col, int *input, int kernel_row,
                  int kernel_col, int *kernel, int output_row, int output_col,
                  long long unsigned int *output) {

  for (int i = 0; i < output_row * output_col; ++i)
    output[i] = 0;

"""

permute_lines = [
"    for (int output_i = 0; output_i < output_row; output_i++) {         // A",
"        for (int output_j = 0; output_j < output_col; output_j++) {     // B",
"          for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++) {   // C",
"            for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++) { // D"
]

source_postamble = """
          int input_i = (output_i + 2 * kernel_i) % input_row;
          int input_j = (output_j + 2 * kernel_j) % input_col;
          output[output_i * output_col + output_j] +=
              input[input_i * input_col + input_j] *
              kernel[kernel_i * kernel_col + kernel_j];
        }
      }
    }
  }
}
"""

def cmd(in_size, kern_size):
    return f'./dilated_conv -i ./data/{in_size}.in -k data/{kern_size}.in'

with open('results.csv', 'w') as results_csv:
    writer = csv.writer(results_csv)
    writer.writerow(['Permutation', *[f'Ref{i}' for i in range(TIMES)], *[f'Opt{i}' for i in range(TIMES)]])
    for permutation in permutations(range(4)):
        p = "".join(["ABCD"[i] for i in permutation])
        entry = [p]
        source = f'//{p}\n' + source_preamble + '\n' + '\n'.join([permute_lines[i] for i in permutation]) + '\n' + source_postamble
        os.system('rm ./header/single_thread.h')
        with open('./header/single_thread.h','w') as header_file:
            header_file.write(source)

        os.system('make')
        ref = []
        opt = []
        for _ in range(TIMES):
            out = os.popen(cmd(IN, KERN)).read()
            print(out)
            out = [x.split()[-2] for x in out.split('\n')[4:6]]
            print(out)
            ref.append(out[0])
            opt.append(out[1])
        entry.extend(ref)
        entry.extend(opt)
        writer.writerow(entry)

