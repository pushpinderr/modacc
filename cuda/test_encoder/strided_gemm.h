#pragma once

#include "gemm.h"
#include <cuda.h>
#include "utils.h"
#include <assert.h>
#include <curand.h>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


template <typename T>
class StridedBatchGemm {
public:
    struct Config {
        int batch_size;
        int m;
        int n;
        int k;
        float alpha;
        float beta;
        cublasOperation_t op_A;
        cublasOperation_t op_B;
        std::array<int, 3> gemm_algos;

        Config(int batch,
               int mm,
               int nn,
               int kk,
               float param_alpha,
               float param_beta,
               cublasOperation_t opA,
               cublasOperation_t opB,
               const std::array<int, 3>& algos)
            : batch_size(batch),
              m(mm),
              n(nn),
              k(kk),
              alpha(param_alpha),
              beta(param_beta),
              op_A(opA),
              op_B(opB),
              gemm_algos(algos)
        {
        }
        void SetConfig(int mm, int nn, int kk)
        {
            m = mm;
            n = nn;
            k = kk;
        }
    };

    StridedBatchGemm(const Config& config) : _config(config) {}

    virtual ~StridedBatchGemm() {}

    void Forward(int bsz, 
                 Buffer<T>* output, 
                 Buffer<T>* _buffer_a, 
                 Buffer<T>* _buffer_b, 
                 ScheduleEngine* SE)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        cublas_strided_batched_gemm(SE->handle,
                                    _config.m,
                                    _config.n,
                                    _config.k,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a->get_device_data(),
                                    _buffer_b->get_device_data(),
                                    output->get_device_data(),
                                    _config.op_A,
                                    _config.op_B,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    bsz,
                                    cublasGemmAlgo_t(_config.gemm_algos[0]));
    }

    /* void ForwardPlusSave(T* output, const T* _buffer_a, const T* _buffer_b, cublasHandle_t handle)
    {
        int stride_a = _config.m * _config.k;
        int stride_b = _config.n * _config.k;
        int stride_c = _config.m * _config.n;

        cublas_strided_batched_gemm(handle,
                                    _config.m,
                                    _config.n,
                                    _config.k,
                                    &_config.alpha,
                                    &_config.beta,
                                    _buffer_a->get_device_data(),
                                    _buffer_b->get_device_data(),
                                    output->get_device_data(),
                                    _config.op_A,
                                    _config.op_B,
                                    stride_a,
                                    stride_b,
                                    stride_c,
                                    _config.batch_size,
                                    cublasGemmAlgo_t(_config.gemm_algos[0]));

        k_buf = _buffer_a;
        q_buf = _buffer_b;
    } */

    inline int GetN() const { return _config.k; }
    inline const T* GetBufferA() const { return k_buf; }
    inline const T* GetBufferB() const { return q_buf; }
    inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

private:
    Config _config;
    const T* q_buf;
    const T* k_buf;
};
