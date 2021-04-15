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

        input_ptr->copyH2D(SE->compute);
        weights->copyH2D(SE->compute);
        // out->copyH2D(SE->compute);

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

        // input_ptr->copyD2H(SE->compute);
        // weights->copyD2H(SE->compute);
        out->copyD2H(SE->compute);      

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

private:
    Config config_;
};