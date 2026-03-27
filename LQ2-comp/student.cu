#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// ─── Error-checking macro ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


/* ============================================================
   Edit ONLY this section.
   ============================================================ */

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

// ─── Naive kernel for N = 64 correctness tests ───────────────────────────────
__global__
void matmul_kernel_naive(const float* __restrict__ A,
                         const float* __restrict__ B,
                               float* __restrict__ C,
                         int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];

    C[row * N + col] = sum;
}

// ─── Ultra-fast Kernel (2D Block Tiling + 2D Register Blocking) ──────────────
// Assumes N is a multiple of 128.
__global__ void __launch_bounds__(256) matmul_fast_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    // Linear thread ID
    int tid = ty * blockDim.x + tx;

    // Allocate shared memory (16 KB total, well under the 32 KB limit)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Thread-local registers for accumulators and operands
    float accum[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // Calculate global memory offsets for loading A and B into shared memory
    // Thread block is 256 threads. Loading 128x16 elements -> 8 elements per thread (2x float4)
    int a_load_row = tid >> 1;        // tid / 2
    int a_load_col = (tid & 1) << 3;  // (tid % 2) * 8
    int a_global_idx = (by * BM + a_load_row) * N + a_load_col;

    // Loading 16x128 elements -> 8 elements per thread (2x float4)
    int b_load_row = tid >> 4;        // tid / 16
    int b_load_col = (tid & 15) << 3; // (tid % 16) * 8
    int b_global_idx = b_load_row * N + bx * BN + b_load_col;

    // Loop over the K dimension
    for (int k = 0; k < N; k += BK) {
        
        // --- Load A tile into Shared Memory (Vectorized) ---
        float4 a_vec1 = *reinterpret_cast<const float4*>(&A[a_global_idx + k]);
        float4 a_vec2 = *reinterpret_cast<const float4*>(&A[a_global_idx + k + 4]);
        As[a_load_row * BK + a_load_col + 0] = a_vec1.x;
        As[a_load_row * BK + a_load_col + 1] = a_vec1.y;
        As[a_load_row * BK + a_load_col + 2] = a_vec1.z;
        As[a_load_row * BK + a_load_col + 3] = a_vec1.w;
        As[a_load_row * BK + a_load_col + 4] = a_vec2.x;
        As[a_load_row * BK + a_load_col + 5] = a_vec2.y;
        As[a_load_row * BK + a_load_col + 6] = a_vec2.z;
        As[a_load_row * BK + a_load_col + 7] = a_vec2.w;

        // --- Load B tile into Shared Memory (Vectorized) ---
        float4 b_vec1 = *reinterpret_cast<const float4*>(&B[b_global_idx + k * N]);
        float4 b_vec2 = *reinterpret_cast<const float4*>(&B[b_global_idx + k * N + 4]);
        Bs[b_load_row * BN + b_load_col + 0] = b_vec1.x;
        Bs[b_load_row * BN + b_load_col + 1] = b_vec1.y;
        Bs[b_load_row * BN + b_load_col + 2] = b_vec1.z;
        Bs[b_load_row * BN + b_load_col + 3] = b_vec1.w;
        Bs[b_load_row * BN + b_load_col + 4] = b_vec2.x;
        Bs[b_load_row * BN + b_load_col + 5] = b_vec2.y;
        Bs[b_load_row * BN + b_load_col + 6] = b_vec2.z;
        Bs[b_load_row * BN + b_load_col + 7] = b_vec2.w;

        __syncthreads();

        // --- Compute Phase ---
        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            
            // Load from shared memory to registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[(ty * TM + i) * BK + dot_idx];
            }

            // Vectorized load from Bs to registers
            float4 b_reg1 = *reinterpret_cast<const float4*>(&Bs[dot_idx * BN + tx * TN + 0]);
            float4 b_reg2 = *reinterpret_cast<const float4*>(&Bs[dot_idx * BN + tx * TN + 4]);
            reg_b[0] = b_reg1.x; reg_b[1] = b_reg1.y; reg_b[2] = b_reg1.z; reg_b[3] = b_reg1.w;
            reg_b[4] = b_reg2.x; reg_b[5] = b_reg2.y; reg_b[6] = b_reg2.z; reg_b[7] = b_reg2.w;

            // FMA (Fused Multiply-Add)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        
        // Sync before loading the next tile over K
        __syncthreads();
    }

    // --- Write back to Global Memory C (Vectorized) ---
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = by * BM + ty * TM + i;
        int global_col = bx * BN + tx * TN;

        float4 c_vec1, c_vec2;
        c_vec1.x = accum[i][0];
        c_vec1.y = accum[i][1];
        c_vec1.z = accum[i][2];
        c_vec1.w = accum[i][3];
        c_vec2.x = accum[i][4];
        c_vec2.y = accum[i][5];
        c_vec2.z = accum[i][6];
        c_vec2.w = accum[i][7];

        *reinterpret_cast<float4*>(&C[global_row * N + global_col + 0]) = c_vec1;
        *reinterpret_cast<float4*>(&C[global_row * N + global_col + 4]) = c_vec2;
    }
}

/**
 * @brief Launch wrapper
 */
void matmul_gpu(int N,
                const float* A_h,
                const float* B_h,
                      float* C_h)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    // ── Allocate device buffers ───────────────────────────────
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));
    CUDA_CHECK(cudaMalloc(&C_d, bytes));

    // ── Transfer inputs to device ─────────────────────────────
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice));

    // ── Launch the Kernel ─────────────────────────────────────
    // If N is a multiple of our 128 block size, run the ultra-fast kernel.
    // Otherwise (e.g., N=64 for early correctness check), use the naive kernel.
    if (N % 128 == 0) {
        dim3 block(16, 16); 
        dim3 grid(N / 128, N / 128);
        matmul_fast_kernel<<<grid, block>>>(A_d, B_d, C_d, N);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y);
        matmul_kernel_naive<<<grid, block>>>(A_d, B_d, C_d, N);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy result back to host ──────────────────────────────
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost));

    // ── Free device memory ────────────────────────────────────
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

/* ============================================================
   END OF STUDENT CODE — do not modify below this line
   ============================================================ */
// ─── CPU reference ────────────────────────────────────────────────────────────
static void matmul_cpu(int N,
                       const float* A,
                       const float* B,
                             float* C)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < N; ++k)
                s += A[i*N+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

// ─── Element-wise verification ────────────────────────────────────────────────
static bool verify(int N, const float* ref, const float* gpu,
                   float tol = 1e-2f)
{
    for (int i = 0; i < N*N; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > tol) {
            int row = i / N, col = i % N;
            fprintf(stderr,
                    "MISMATCH at (%d,%d): ref=%.6f  gpu=%.6f  |diff|=%.2e\n",
                    row, col, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main()
{
    // ── Correctness tests (small sizes, CPU reference) ────────
    printf("=== Correctness Tests ===\n");
    {
        const std::vector<int> small_sizes = {64, 128, 256, 512};
        bool all_ok = true;

        for (int N : small_sizes) {

            std::vector<float> A(N*N), B(N*N),
                               C_cpu(N*N, 0.f),
                               C_gpu(N*N, 0.f);

            for (int i = 0; i < N*N; ++i) {
                A[i] = (float)(i % 97) / 97.f;
                B[i] = (float)((i * 7 + 3) % 97) / 97.f;
            }

            matmul_cpu(N, A.data(), B.data(), C_cpu.data());
            matmul_gpu(N, A.data(), B.data(), C_gpu.data());

            bool ok = verify(N, C_cpu.data(), C_gpu.data());
            printf("  N = %4d : %s\n", N, ok ? "PASSED" : "FAILED");
            all_ok &= ok;
        }

        if (!all_ok) {
            fprintf(stderr,
                    "\nCorrectness FAILED — fix your kernel before optimising.\n");
            return EXIT_FAILURE;
        }
        printf("All correctness tests PASSED.\n\n");
    }

    return EXIT_SUCCESS;
}
