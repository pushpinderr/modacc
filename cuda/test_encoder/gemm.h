#pragma once

#include <cuda.h>
#include <time.h>
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define THREADS 256
#define TILE_DIM 32

#define minus_infinity -1 * std::numeric_limits<float>::infinity()

#define FINAL_MASK 0xffffffff

bool useMean = true;

__global__ void fused_add2_kernel(const int N, float* out, const float* inp1, const float* inp2)
{
    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    float4* out_4 = reinterpret_cast<float4*>(out);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        float4 val;
        float4 inp1_reg = inp1_4[j];
        float4 inp2_reg = inp2_4[j];

        val.x = inp1_reg.x + inp2_reg.x;
        val.y = inp1_reg.y + inp2_reg.y;
        val.z = inp1_reg.z + inp2_reg.z;
        val.w = inp1_reg.w + inp2_reg.w;

        out_4[j] = val;
    }
}

template <typename T>
void launch_fused_add2(T* out,
                        const T* inp1,
                        const T* inp2,
                        int batch_size,
                        int seq_length,
                        int hidden_dim,
                        cudaStream_t stream);

template <>
void launch_fused_add2(float* out,
                              const float* inp1,
                              const float* inp2,
                              int batch_size,
                              int seq_length,
                              int hidden_dim,
                              cudaStream_t stream)
{
    int total_count = batch_size * seq_length * hidden_dim / 4;
    dim3 grid_dim = DS_GET_BLOCKS(total_count);  //(batch_size * seq_length);

    dim3 block_dim = DS_CUDA_NUM_THREADS;  //(hidden_dim / 4);

    fused_add2_kernel<<<grid_dim, block_dim, 0, stream>>>(total_count, out, inp1, inp2);
}

int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         CUDA_R_32F,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         CUDA_R_32F,
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         C,
                                         CUDA_R_32F,
                                         m,
                                         CUDA_R_32F,
                                         algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n", m, n, k, (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         CUDA_R_16F,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         CUDA_R_16F,
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         (void*)C,
                                         CUDA_R_16F,
                                         m,
                                         CUDA_R_32F,
                                         algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n", m, n, k, (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       CUDA_R_32F,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       CUDA_R_32F,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       CUDA_R_32F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n", batch, m, n, k, (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       CUDA_R_16F,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       CUDA_R_16F,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       CUDA_R_16F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n", m, n, k, (int)status);
        return EXIT_FAILURE;
    }

    return 0;
}

template <typename T>
int cublas_fine_gemm_ex(const T* input_ptr,
                        const T* weights,
                        T* out,
                        int outputSize,
                        int bsz,
                        int inputSize,
                        cublasHandle_t handle,
                        cudaStream_t* stream,
                        int q_index,
                        cublasGemmAlgo_t algo)
{
    float alpha = T(1.);
    float beta = T(0.);
    cublasSetStream(handle, stream[q_index]);

    return cublas_gemm_ex(handle,
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          outputSize,
                          bsz,
                          inputSize,
                          &alpha,
                          &beta,
                          weights,
                          input_ptr,
                          out,
                          algo);
}

template <typename T>
void launch_fuse_transpose_bias_kernel(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols);

template <typename T>
__global__ void column_sum_reduce(const T* __restrict__ inp,
                                  T* __restrict__ out,
                                  int rows,
                                  int width);

template <>
void launch_fuse_transpose_bias_kernel<float>(const float* inp,
                                              float* out,
                                              int rows,
                                              int cols)
{
    dim3 grid_dim((cols - 1) / TILE_DIM + 1);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    column_sum_reduce<float><<<grid_dim, block_dim, 0>>>(inp, out, rows, cols);
}

template <typename T>
__global__ void column_sum_reduce(const T* __restrict__ inp,
                                  T* __restrict__ out,
                                  int rows,
                                  int width)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int y_stride = width * TILE_DIM;

    float localSum = 0;

    if (idx < width) {
        int offset = threadIdx.y * width + idx;
        for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
            localSum += (float)inp[offset];
            offset += y_stride;
        }
    }

    tile[threadIdx.x][threadIdx.y] = localSum;

    __syncthreads();

    float sum = tile[threadIdx.y][threadIdx.x];

    #ifndef __STOCHASTIC_MODE__
    __syncthreads();
    #endif

    for (int i = 1; i < TILE_DIM; i <<= 1) sum += g.shfl_down(sum, i);

    if (threadIdx.x == 0) {
        int pos = blockIdx.x * TILE_DIM + threadIdx.y;
        if (pos < width) out[pos] = sum;
    }
}

template <typename T>
void launch_fuse_transpose_bias_kernel(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols,
                                       cudaStream_t stream);

template <>
void launch_fuse_transpose_bias_kernel(const float* inp,
                                              float* out,
                                              int rows,
                                              int cols,
                                              cudaStream_t stream)
{
    // assert(rows % TILE_DIM == 0);
    // assert(cols % TILE_DIM == 0);

    dim3 grid_dim((cols - 1) / TILE_DIM + 1);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    column_sum_reduce<float><<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

template <typename T>
class FeedForward {
public:
    struct Config {
        int batchSize; 
        int outputSize;
        int inputSize;
        std::array<int, 3> gemm_algos;
        bool training;
        Config(int batch, int outputs, int inputs, const std::array<int, 3>& algos, bool training) 
            : batchSize(batch), outputSize(outputs), inputSize(inputs), gemm_algos(algos), training(training)
        {
        }
    };

    FeedForward(Config config) : config_(config) {}

    ~FeedForward() {}

    void ForwardCheckpoint(int bsz,  // batch * seq
                           Buffer<T>* input_ptr,
                           Buffer<T>* weights,
                           Buffer<T>* out,
                           ScheduleEngine* SE,
                           int sync=false)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        input_ptr->copyH2D(SE->compute);
        weights->copyH2D(SE->compute);
        // out->copyH2D(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
	        printf("H2D Time:%lf\n",sw.GetTimeInSeconds());
	        sw.restart();
        #endif

        cublas_fine_gemm_ex(input_ptr->get_device_data(),
                            weights->get_device_data(),
                            out->get_device_data(),
                            config_.outputSize,
                            bsz,
                            config_.inputSize,
                            SE->handle,
                            SE->compute,
                            0,
                            cublasGemmAlgo_t(config_.gemm_algos[0]));

        if ( sync == true )
            CHECK(cudaThreadSynchronize());

        #if EVENT_PROFILE
	        sw.stop();
	        printf("Kernel Time:%lf\n",sw.GetTimeInSeconds());
	        sw.restart();
        #endif

        // input_ptr->copyD2H(SE->compute);
        // weights->copyD2H(SE->compute);
        out->copyD2H(SE->compute);

        #if EVENT_PROFILE
	        sw.stop();
	        printf("D2H Time:%lf\n",sw.GetTimeInSeconds());
	        sw.restart();
        #endif
    }

    void ForwardCheckpointPartition(int bsz,  // batch * seq
                                    Buffer<T>* input_ptr,
                                    Buffer<T>* weights,
                                    Buffer<T>* out,
                                    ScheduleEngine* SE,
                                    int nq,
                                    int sync=true)
    {
        weights->copyH2D(SE->compute);
        int offset = 0;
        int offset_size = bsz * config_.inputSize / nq;

        #if DEBUG
            std::cout << "offset_size=" << offset_size << std::endl;
            std::cout << "input volume=" << bsz*config_.inputSize << std::endl;
            std::cout << "output volume=" << 3*bsz*config_.inputSize << std::endl;
        #endif

        for (int i = 0; i < nq; i++)
        {
            offset = i * offset_size;   
            input_ptr->copyH2D(SE->compute, offset, nq, i);
            // out->copyH2D(SE->compute, offset, nq, i);

            #if DEBUG
                std::cout << "\x1b[31;1mqueue index=" << i << "\x1b[0m" << std::endl;
                std::cout << "input offset=" << offset << std::endl;
                std::cout << "output offset=" << 3*offset << std::endl;
            #endif

            cublasSetStream(SE->handle, SE->compute[i]);
            cublas_fine_gemm_ex(input_ptr->get_device_data(offset),
                                weights->get_device_data(),
                                out->get_device_data(3*offset),
                                config_.outputSize,
                                bsz / nq,
                                config_.inputSize,
                                SE->handle,
                                SE->compute,
                                i,
                                cublasGemmAlgo_t(config_.gemm_algos[0]));

            // input_ptr->copyD2H(SE->compute, offset, nq, i);
            out->copyD2H(SE->compute, offset, nq, i);      
        }
     
        if ( sync == true )
            CHECK(cudaThreadSynchronize());
    }

    void Backward(int bsz,
                  Buffer<T>* out_grad,
                  Buffer<T>* input_ptr,
                  Buffer<T>* weights,
                  Buffer<T>* weights_grad,
                  Buffer<T>* bias_grad,
                  ScheduleEngine* SE,
                  Buffer<T>* inp_grad_out = nullptr)
    {
        float alpha = (T)1.0, beta = (T)0.0;
        cublas_gemm_ex(SE->handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       config_.inputSize,
                       config_.outputSize,
                       bsz,
                       &alpha,
                       &beta,
                       input_ptr->get_device_data(),
                       out_grad->get_device_data(),
                       weights_grad->get_device_data(),
                       cublasGemmAlgo_t(config_.gemm_algos[1]));

        cublas_gemm_ex(SE->handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       config_.inputSize,
                       bsz,
                       config_.outputSize,
                       &alpha,
                       &beta,
                       weights->get_device_data(),
                       out_grad->get_device_data(),
                       inp_grad_out->get_device_data(),
                       cublasGemmAlgo_t(config_.gemm_algos[2]));

        launch_fuse_transpose_bias_kernel(out_grad->get_device_data(), bias_grad->get_device_data(), bsz, config_.outputSize, SE->getStream(0));
    }

    void BackwardFineGrained(int bsz,
                            int nq,
                            Buffer<T>* out_grad,
                            Buffer<T>* input_ptr,
                            Buffer<T>* weights,
                            Buffer<T>* weights_grad,
                            Buffer<T>* bias_grad,
                            ScheduleEngine* SE,
                            Buffer<T>* inp_grad_out = nullptr)
    {
        float alpha = (T)1.0, beta = (T)0.0;

        weights->copyH2D(SE->compute);
        weights_grad->copyH2D(SE->compute);
        bias_grad->copyH2D(SE->compute);

        int offset = 0;
        int offset_size = bsz * config_.inputSize / nq;

        #if DEBUG
            std::cout << "offset_size=" << offset_size << std::endl;
            std::cout << "input volume=" << bsz*config_.inputSize << std::endl;
            std::cout << "output volume=" << 3*bsz*config_.inputSize << std::endl;
        #endif

        for (int i = 0; i < nq; i++)
        {
            offset = i * offset_size;   
            input_ptr->copyH2D(SE->compute, offset, nq, i);

            #if DEBUG
                std::cout << "\x1b[31;1mqueue index=" << i << "\x1b[0m" << std::endl;
                std::cout << "input offset=" << offset << std::endl;
                std::cout << "output offset=" << 3*offset << std::endl;
            #endif

            cublasSetStream(SE->handle, SE->compute[i]);

            cublas_gemm_ex(SE->handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       config_.inputSize,
                       config_.outputSize,
                       bsz/nq,
                       &alpha,
                       &beta,
                       input_ptr->get_device_data(offset),
                       out_grad->get_device_data(3*offset),
                       weights_grad->get_device_data(),
                       cublasGemmAlgo_t(config_.gemm_algos[1]));

            cublas_gemm_ex(SE->handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       config_.inputSize,
                       bsz/nq,
                       config_.outputSize,
                       &alpha,
                       &beta,
                       weights->get_device_data(),
                       out_grad->get_device_data(3*offset),
                       inp_grad_out->get_device_data(offset),
                       cublasGemmAlgo_t(config_.gemm_algos[2]));

            launch_fuse_transpose_bias_kernel(out_grad->get_device_data(), bias_grad->get_device_data(), bsz, config_.outputSize, SE->getStream(0));

            out_grad->copyD2H(SE->compute, offset, nq, i);      
        }
    }

    private:
        Config config_;
};