#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

__global__ void matmul_gpu_coalesced(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int N = atoi(argv[1]);

    size_t size = N*N*sizeof(float);
    float *A, *B, *C;
    float *dA, *dB, *dC;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for (int i = 0; i < N*N; ++i) {
        A[i] = 1.0f; 
        B[i] = 1.0f; 
    }

    CUDA_CHECK(cudaMalloc(&dA, size));
    CUDA_CHECK(cudaMalloc(&dB, size));
    CUDA_CHECK(cudaMalloc(&dC, size));

    CUDA_CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEventRecord(start, stream);
    matmul_gpu_coalesced<<<blocks, threads, 0, stream>>>(dA, dB, dC, N);
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost));

    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Kernel Execution Time: "<<time<<" ms"<<endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);


    free(A);
    free(B);
    free(C);
}