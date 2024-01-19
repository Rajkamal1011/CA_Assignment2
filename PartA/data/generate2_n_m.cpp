#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char *argv[])
{
    int N; 
    int M;
    if (argc < 3) {
        N = 64;
        M = 64;
    }
    else{
        N = atoi(argv[1]); 
        M = atoi(argv[2]);
    }
    ofstream output_file; 
    string file_name = "data/" + to_string(N) +"_"+to_string(M)+ ".in";  
    output_file.open(file_name); 

    // Set seed
    srand(1);
    
    // Output row and column of the matrix size
    output_file << N << "\n";
    output_file << M << "\n"; 
    // Generate matrix A
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j)
            output_file << rand() % 256 << " ";
        output_file << "\n";
    }
    output_file.close(); 
    return 0; 
}
