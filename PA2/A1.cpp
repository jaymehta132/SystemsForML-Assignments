#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

void matmul_naive_cpu(float* A, float* B, float* C, int N) {
    for (int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            float sum = 0.0f;
            for(int k=0; k<N; k++){
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(float);
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    // Initialize A and B with all 1s
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    // auto start = chrono::high_resolution_clock::now();
    matmul_naive_cpu(A, B, C, N);
    // auto end = chrono::high_resolution_clock::now();

    // double duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    // cout << "Time taken for matrix multiplication: " << duration << " ms" << endl;

    free(A);
    free(B);
    free(C);
}