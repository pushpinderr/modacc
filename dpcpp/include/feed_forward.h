#ifndef __FEEDFORWARD_H__
#define __FEEDFORWARD_H__

#include <CL/sycl.hpp>
#include <stdio.h>
using namespace cl::sycl;

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

template <typename T>

class FeedForward {
  public:
    struct Config {
        int batchSize, outputSize;
        int inputSize;

        Config(int batch, int inputs, int outputs)
            : batchSize(batch), outputSize(outputs), inputSize(inputs) {}
    };

    FeedForward(Config config) : config_(config) {}

    ~FeedForward() {}

    void ForwardPartitionWeights(int bsz, const T *input_ptr, const T *weights,
                                 T *out, std::vector<queue> &_queues,
                                 int num_queues) {

        float alpha = T(1.);
        float beta = T(0.);
        auto transA = oneapi::mkl::transpose::trans;
        auto transB = oneapi::mkl::transpose::nontrans;
        int m = config_.outputSize;
        int k = config_.inputSize;
        int n = config_.batchSize;
        int lda = (transA == oneapi::mkl::transpose::nontrans;) ? m : k;
        int ldb = (transB == oneapi::mkl::transpose::nontrans;) ? k : n;
        int ldc = m;
        int granularity = num_queues;
        int offset = 0;
        for (int i = 0; i < num_queues; i++) {
            auto ex = oneapi::mkl::blas::row_major::gemm(
                _queues[i], transA, transB, m / granularity, n, k, alpha,
                weights + offset * k, lda, input_ptr, ldb, beta,
                out + offset * n, ldc, {h2d});
            offset += m / granularity;
        }

        /*
        void Backward(int bsz,
                      const T* out_grad,
                      const T* input_ptr,
                      const T* weights,
                      T* weights_grad,
                      T* bias_grad,
                      cublasHandle_t& _cublasHandle,
                      cudaStream_t& stream,
                      T* inp_grad_out = nullptr,
                      T* out_grad_trans_out = nullptr)
        {
            float alpha = (T)1.0, beta = (T)0.0;
            cublas_gemm_ex(_cublasHandle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           config_.inputSize,
                           config_.outputSize,
                           bsz,
                           &alpha,
                           &beta,
                           input_ptr,
                           out_grad,
                           weights_grad,
                           cublasGemmAlgo_t(config_.gemm_algos[1]));

            cublas_gemm_ex(_cublasHandle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           config_.inputSize,
                           bsz,
                           config_.outputSize,
                           &alpha,
                           &beta,
                           weights,
                           out_grad,
                           inp_grad_out,
                           cublasGemmAlgo_t(config_.gemm_algos[2]));

            launch_fuse_transpose_bias_kernel<T>(out_grad, bias_grad, bsz,
        config_.outputSize, stream);
        }
    */
      private:
        Config config_;
    };

#endif
