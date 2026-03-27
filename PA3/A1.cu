#include <iostream>
#include <cuda_runtime.h>
#include <random>

using namespace std;

#define batchSize 32 

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


// matmul of M x N with N x P (A -> M x N, B -> N x P, C -> M x P)
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int i = 0; i < K; i++)
            sum += A[row * K + i] * B[i * N + col];

        C[row * N + col] = sum;
    }
}

// relu function on M x N (A -> M x N, B -> M x N)
__global__ void relu(float* A, float* B, int M, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if (row < M && col < N){
        B[row * N + col] = fmaxf(A[row * N + col], 0.0f);
    }
}

int main(int argc, char** argv){
    int N = atoi(argv[1]);

    float *X, *W1, *W2, *Z;
    float *dX, *dW1, *dW2, *dZ, *dY, *dZ1;

    size_t size_X = batchSize * N * sizeof(float);      // X -> N x B
    size_t size_W = N * N * sizeof(float);              // W1, W2 -> N x N
    size_t size_Z = batchSize * N * sizeof(float);      // Z -> N x B

    X = (float*)malloc(size_X);
    W1 = (float*)malloc(size_W);
    W2 = (float*)malloc(size_W);
    Z = (float*)malloc(size_Z);

    CUDA_CHECK(cudaMalloc(&dX, size_X));
    CUDA_CHECK(cudaMalloc(&dW1, size_W));
    CUDA_CHECK(cudaMalloc(&dW2, size_W));
    CUDA_CHECK(cudaMalloc(&dZ, size_Z));
    CUDA_CHECK(cudaMalloc(&dY, size_X));    // Y = ReLU(W1 * X)
    CUDA_CHECK(cudaMalloc(&dZ1, size_X));   // Z1 = W1 * X

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < batchSize * N; i++){
        X[i] = dist(gen);
    }
    for (int i = 0; i < N * N; i++){
        W1[i] = dist(gen);
        W2[i] = dist(gen);
    }

    CUDA_CHECK(cudaMemcpy(dX, X, size_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, W1, size_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2, size_W, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(32, 32);
    dim3 blocks((batchSize + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);
    cudaEventRecord(start, stream);

    matmul<<<blocks, threads, 0, stream>>>(dW1, dX, dZ1, N, N, batchSize);
    CUDA_CHECK(cudaGetLastError());

    relu<<<blocks, threads, 0, stream>>>(dZ1, dY, N, batchSize);
    CUDA_CHECK(cudaGetLastError());

    matmul<<<blocks, threads, 0, stream>>>(dW2, dY, dZ, N, N, batchSize);
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CU
    CUDA_CHECK(cudaMemcpy(Z, dZ, size_Z, cudaMemcpyDeviceToHost));

    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Kernel Execution Time: "<<time<<" ms"<<endl;

    cudaFree(dX);
    cudaFree(dW1);
    cudaFree(dW2);
    cudaFree(dY);
    cudaFree(dZ1);
    cudaFree(dZ);

    free(X);
    free(W1);
    free(W2);
    free(Z);
}

