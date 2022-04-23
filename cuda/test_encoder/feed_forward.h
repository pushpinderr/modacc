#ifndef __FEEDFORWARD_H__
#define __FEEDFORWARD_H__

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

template <typename T>
class FeedForward {
public:
    struct Config {
        int batchSize, outputSize;
        int inputSize;
        std::array<int, 3> gemm_algos;
        Config(int batch, int outputs, int inputs, const std::array<int, 3>& algos)
            : batchSize(batch), outputSize(outputs), inputSize(inputs), gemm_algos(algos)
        {
        }
    };

    FeedForward(Config config) : config_(config) {}

    ~FeedForward() {}

    void Forward(int bsz,
                 Buffer <T>* input_ptr,
                 Buffer <T>* weights,
                 Buffer <T>* out,
                 ScheduleEngine* SE)
    {
        float alpha = T(1.);
        float beta = T(0.);

        cublas_gemm_ex(SE->handle,
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       config_.outputSize,
                       bsz,
                       config_.inputSize,
                       &alpha,
                       &beta,
                       weights,
                       input_ptr,
                       out,
                       cublasGemmAlgo_t(config_.gemm_algos[0]));
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
                       input_ptr,
                       out_grad,
                       weights_grad,
                       cublasGemmAlgo_t(config_.gemm_algos[1]));

        cublas_gemm_ex(SE->handle,
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

        launch_fuse_transpose_bias_kernel<T>(out_grad, bias_grad, bsz, config_.outputSize, SE->getStream(0));
    }

    template <typename T>
    void launch_fuse_transpose_bias_kernel(const T* inp,
                                        T* out,
                                        int rows,
                                        int cols,
                                        cudaStream_t stream);    

private:
    Config config_;
};

#endif
