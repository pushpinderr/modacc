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

template <typename T>
void launch_fused_add2(T* out,
                       const T* inp1,
                       const T* inp2,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       cudaStream_t& stream);

template <typename T>
void launch_fused_add4(T* out,
                       const T* inp1,
                       const T* inp2,
                       const T* inp3,
                       const T* inp4,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       cudaStream_t& stream);

template <typename T>
void launch_fused_add3(T* out,
                       const T* inp1,
                       const T* inp2,
                       const T* inp3,
                       int batch_size,
                       int seq_length,
                       int hidden_size,
                       cudaStream_t& stream);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* X_data,
                               const T* vars,
                               const T* means,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               cudaStream_t stream[2]);

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                               const T* vals_hat,
                               const T* vars,
                               const T* gamma,
                               T* gamma_grad,
                               T* betta_grad,
                               T* inp_grad,
                               int batch_size,
                               int hidden_dim,
                               cudaStream_t stream[2],
                               bool invertible = false,
                               const T* betta = nullptr);

inline bool UseMean()
{
    return useMean;
}

template <typename T>
void launch_dropout_grad(T* vals, uint8_t* mask, int total_count, float ratio, cudaStream_t stream);

template <typename T>
void launch_dropout_grad(T* vals_out,
                         const T* vals,
                         uint8_t* mask,
                         int total_count,
                         float ratio,
                         cudaStream_t stream);
template <typename T>
void Backward(int bsz, T* d_vals, cudaStream_t stream)
{
    launch_dropout_grad<T>(d_vals, _mask, bsz * _config.dim, _config.RATIO(), stream);
}

template <typename T>
void Backward(int bsz, T* d_vals_out, const T* d_vals, cudaStream_t stream)
{
    launch_dropout_grad<T>(d_vals_out, d_vals, _mask, bsz * _config.dim, _config.RATIO(), stream);
}

bool HasDropout()
{ 
    return _config.RATIO() > 0.0;
}

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     cudaStream_t stream);

template <typename T>
void Backward(int bsz, T* out_grad, const T* soft_out, cudaStream_t stream)
{
    launch_attn_softmax_backward_v2<T>(out_grad, soft_out, bsz, config_.heads, config_.seq_length, stream);
}

template <typename T>
void Forward(int bsz, T* out, const T* vals, cudaStream_t stream, bool bwd = false)
{
    launch_dropout<T>(out, vals, _mask, bsz * _config.dim, _config.dim, _config.RATIO(), stream, bwd);
}

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* X_data,
                                         const T* vars,
                                         const T* means,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         cudaStream_t stream[2]);
template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                         const T* out_grad2,
                                         const T* vals_hat,
                                         const T* vars,
                                         const T* gamma,
                                         T* gamma_grad,
                                         T* betta_grad,
                                         T* inp_grad,
                                         int batch_size,
                                         int hidden_dim,
                                         cudaStream_t stream[2],
                                         bool invertible = false,
                                         const T* betta = nullptr);
template <typename T>
void BackwardFusedAdd(int bsz,
                          const T* out_grad1,
                          const T* out_grad2,
                          const T* gamma,
                          T* gamma_grad,
                          T* betta_grad,
                          cudaStream_t stream[2],
                          T* inp_grad_out,
                          const T* norm_in = nullptr)
    {
        launch_layerNorm_backward_fused_add(out_grad1,
                                            out_grad2,
                                            norm_in,
                                            vars,
                                            means,
                                            gamma,
                                            gamma_grad,
                                            betta_grad,
                                            inp_grad_out,
                                            bsz,
                                            config_.hiddenDim,
                                            stream);
    }

template <typename T>
void BackwardFusedAdd(int bsz,
                          const T* out_grad1,
                          const T* out_grad2,
                          const T* gamma,
                          const T* betta,
                          T* gamma_grad,
                          T* betta_grad,
                          cudaStream_t stream[2],
                          T* inp_grad_out,
                          const T* norm_out)
    {
        launch_layerNorm_backward_fused_add(out_grad1,
                                            out_grad2,
                                            norm_out,
                                            vars,
                                            gamma,
                                            gamma_grad,
                                            betta_grad,
                                            inp_grad_out,
                                            bsz,
                                            config_.hiddenDim,
                                            stream,
                                            !config_.useMean,
                                            betta);
    }

template <typename T>
void launch_d_gelu(T* d_output,
                   const T* input,
                   const T* bias,
                   int intermediate_size,
                   int batch_size,
                   cudaStream_t stream);
template <typename T>
void Backward(int bsz, T* d_output, const T* input_buf, const T* bias, cudaStream_t stream)
{
    launch_d_gelu<T>(d_output, input_buf, bias, _config.intermediate_size, bsz, stream);
}

// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(const T* input,
                      const T* bias,
                      T* output,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);

template <typename T>
void ForwardWithBiasAdd(int bsz,
                            const T* input_buf,
                            const T* bias,
                            T* output,
                            cudaStream_t stream)
{
    launch_bias_gelu<T>(input_buf, bias, output, _config.intermediate_size, bsz, stream);
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
                                       int cols
                                       );

template <typename T>
__global__ void column_sum_reduce(const T* __restrict__ inp,
                                  T* __restrict__ out,
                                  int rows,
                                  int width);

template <>
void launch_fuse_transpose_bias_kernel<float>(const float* inp,
                                              float* out,
                                              int rows,
                                              int cols
                                              )
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
                  int sync,
                  ScheduleEngine* SE,
                  
                  Buffer<T>* inp_grad_out = nullptr,
                  Buffer<T>* out_grad_trans_out = nullptr)
    {
        float alpha = (T)1.0, beta = (T)0.0;
       // cublas_gemm_ex(SE,
                       //CUBLAS_OP_N,
                       //CUBLAS_OP_T,
           //            config_.inputSize,
         //              config_.outputSize,
             //          bsz,
                       //&alpha,
                       //&beta,
               //        input_ptr->get_device_data(),
                 //      out_grad->get_device_data(),
                   //    weights_grad->get_device_data(),
                     //  cublasGemmAlgo_t(config_.gemm_algos[1]));
	cublas_gemm_ex(input_ptr->get_device_data(), 
		weights_grad->get_device_data(),
		out_grad->get_device_data(),
		config_.outputSize,
		bsz,
		config_.inputSize,
		SE->handle,
		SE->compute,
		0,
		cublasGemmAlgo_t(config_.gemm_algos[0]));


        //cublas_gemm_ex(SE,
                       //CUBLAS_OP_N,
                       //CUBLAS_OP_N,
          //             config_.inputSize,
            //           bsz,
              //         config_.outputSize,
                       //&alpha,
                       //&beta,
                //       weights->get_device_data(),
                  //     out_grad->get_device_data(),
                    //   inp_grad_out->get_device_data(),
                      // cublasGemmAlgo_t(config_.gemm_algos[2]));

cublas_gemm_ex(input_ptr->get_device_data(), 
		weights_grad->get_device_data(),
		out_grad->get_device_data(),
		config_.outputSize,
		bsz,
		config_.inputSize,
		SE->handle,
		SE->compute,
		0,
		cublasGemmAlgo_t(config_.gemm_algos[0]));

        launch_fuse_transpose_bias_kernel<T>(out_grad, bias_grad, bsz, config_.outputSize);
    }

void Backward(int bsz,
                  const T* out_grad,
                  const T* gamma,
                  T* gamma_grad,
                  T* betta_grad,
                  cudaStream_t stream[2],
                  T* inp_grad_out,
                  const T* norm_in = nullptr)
    {
        launch_layerNorm_backward(out_grad,
                                  norm_in,
                                  vars,
                                  means,
                                  gamma,
                                  gamma_grad,
                                  betta_grad,
                                  inp_grad_out,
                                  bsz,
                                  config_.hiddenDim,
                                  stream);
    }

    void Backward(int bsz,
                  const T* out_grad,
                  const T* gamma,
                  const T* betta,
                  T* gamma_grad,
                  T* betta_grad,
                  cudaStream_t stream[2],
                  T* inp_grad_out,
                  const T* norm_out)
    {
        launch_layerNorm_backward(out_grad,
                                  norm_out,
                                  vars,
                                  gamma,
                                  gamma_grad,
                                  betta_grad,
                                  inp_grad_out,
                                  bsz,
                                  config_.hiddenDim,
                                  stream,
                                  !config_.useMean,
                                  betta);
    }

private:
    Config config_;
};
