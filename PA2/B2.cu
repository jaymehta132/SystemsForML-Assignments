#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_DIM 32
#define STRIDE 4

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

__global__ void matmul_tiled_rowbased(float* A, float* B, float* C, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col_start = bx * TILE_DIM * STRIDE + tx;

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM * STRIDE];

    float sums[STRIDE] = {0.0f};
    for (int tile = 0; tile < (N + TILE_DIM - 1)/(TILE_DIM); tile++){
        int aCol = tile * TILE_DIM + ty;
        int bRow = tile * TILE_DIM + tx;
        sA[ty][tx] = A[row * N + aCol];

        for (int e = 0; e < STRIDE; e++){
            sB[ty][tx + e * TILE_DIM] = B[bRow * N + (col_start + e * TILE_DIM)];
        }
        __syncthreads();
        for (int k = 0; k < TILE_DIM; k++){
            for (int e = 0; e <  STRIDE; e++){
                sums[e] += sA[ty][k] * sB[k][tx + e * TILE_DIM];
            }
        }
        __syncthreads();
    }
    for (int e = 0; e < STRIDE; e++){
        C[row * N + (col_start + e * TILE_DIM)] = sums[e];
    }
}

int main(int argc, char** argv){
    int N = atoi(argv[1]);

    size_t size = N * N * sizeof(float);

    
    float *A, *B, *C, *dA, *dB, *dC;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);
    
    CUDA_CHECK(cudaMalloc(&dA, size));
    CUDA_CHECK(cudaMalloc(&dB, size));
    CUDA_CHECK(cudaMalloc(&dC, size));
    
    for (int i = 0; i< N * N; i++){
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    
    CUDA_CHECK(cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM * STRIDE - 1)/(TILE_DIM * STRIDE), (N + TILE_DIM - 1)/TILE_DIM);

    cudaEventRecord(start, stream);
    matmul_tiled_rowbased<<<blocks, threads, 0, stream>>>(dA, dB, dC, N);
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

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}