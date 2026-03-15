#include <iostream>
#include <cuda_runtime.h>

// using namespace std;

#define TILE_WIDTH 32

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

__global__ void matmul_tiled_single(float* A, float* B, float* C, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    
    float val = 0.0f;
    for (int phase = 0; phase < ((N + TILE_WIDTH - 1) / TILE_WIDTH); phase++){
        if ((row < N) && (phase * TILE_WIDTH + tx) <  N){
            sA[ty][tx] = A[row * N + (phase * TILE_WIDTH) + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if ((phase * TILE_WIDTH + ty < N) && col < N){
            sB[ty][tx] = B[((phase * TILE_WIDTH) + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++){
            val += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N){
        C[row * N + col] = val;
    }

}

int main(int argc, char** argv){
    int N = atoi(argv[1]);
    size_t size = N * N * sizeof(float);

    float *A, *B, *C;
    float *dA, *dB, *dC;
    // CUDA_CHECK(cudaMallocHost(&A, size));
    A = (float*)malloc(size);
    // CUDA_CHECK(cudaMallocHost(&B, size));
    B = (float*)malloc(size);
    // CUDA_CHECK(cudaMallocHost(&C, size));
    C = (float*)malloc(size);

    CUDA_CHECK(cudaMalloc(&dA, size));
    CUDA_CHECK(cudaMalloc(&dB, size));
    CUDA_CHECK(cudaMalloc(&dC, size));

    for (int i = 0; i < N * N; i++){
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    CUDA_CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));

    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    matmul_tiled_single<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}