#pragma once

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

inline __device__ float gelu(const float x) // gelu activation function.
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

inline __device__ float d_gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;

    float x2mul = x * x * mul_param;
    float tan_h = tanhf(sqrt_param * (x + x * x2mul));
    float dg1 = 0.5f * (1.0f + tan_h);
    float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
    float dg3 = dg2 * 3 * x2mul;
    return (dg1 + dg2 + dg3);
}

__global__ void fused_bias_gelu(const float* input,
                                const float* bias,
                                float* vals,
                                int row_stride,
                                int iterations)
{
    int row = blockIdx.x;
    int id = threadIdx.x;
    int loop_stride = blockDim.x;

    const float4* input_cast = reinterpret_cast<const float4*>(input);
    float4* vals_cast = reinterpret_cast<float4*>(vals);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float4 data = input_cast[row * row_stride + i * loop_stride + id];
            float4 bias_data = bias_cast[i * loop_stride + id];

            data.x += bias_data.x;
            data.y += bias_data.y;
            data.z += bias_data.z;
            data.w += bias_data.w;

            data.x = gelu(data.x);
            data.y = gelu(data.y);
            data.z = gelu(data.z);
            data.w = gelu(data.w);

            vals_cast[row * row_stride + i * loop_stride + id] = data;
        }
    }
}

__global__ void fused_bias_gelu(const __half* input,
                                const __half* bias,
                                __half* vals,
                                int row_stride,
                                int iterations)
{
    #if __CUDA_ARCH__ >= 700
        int row = blockIdx.x;
        int id = threadIdx.x;
        int loop_stride = blockDim.x;

        const float2* input_cast = reinterpret_cast<const float2*>(input);
        float2* vals_cast = reinterpret_cast<float2*>(vals);
        const float2* bias_cast = reinterpret_cast<const float2*>(bias);

        for (int i = 0; i < iterations; i++) {
            if (i * loop_stride + id < row_stride) {
                float2 vals_vec = input_cast[row * row_stride + i * loop_stride + id];
                float2 bias_vec = bias_cast[i * loop_stride + id];

                __half2* vals_half = reinterpret_cast<__half2*>(&vals_vec);
                __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);

                float2 low_data = __half22float2(vals_half[0]);
                float2 high_data = __half22float2(vals_half[1]);

                float2 low_bias = __half22float2(bias_half[0]);
                float2 high_bias = __half22float2(bias_half[1]);

                low_data.x += low_bias.x;
                low_data.y += low_bias.y;
                high_data.x += high_bias.x;
                high_data.y += high_bias.y;

                low_data.x = gelu(low_data.x);
                low_data.y = gelu(low_data.y);
                high_data.x = gelu(high_data.x);
                high_data.y = gelu(high_data.y);

                vals_half[0] = __float22half2_rn(low_data);
                vals_half[1] = __float22half2_rn(high_data);

                vals_cast[row * row_stride + i * loop_stride + id] = vals_vec;
            }
        }
    #endif
}

template <typename T>
void launch_bias_gelu(const T* input,
                      const T* bias,
                      T* output,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations * 4) + 1;
    dim3 block_dims(threads);
    dim3 grid_dims(batch_size);

    fused_bias_gelu<<<grid_dims, block_dims, 0, stream>>>(
        input, bias, output, intermediate_size / 4, iterations);
}

template void launch_bias_gelu<float>(const float*,
                                    const float*,
                                    float*,
                                    int,
                                    int,
                                    cudaStream_t);

template void launch_bias_gelu<__half>(const __half*,
                                    const __half*,
                                    __half*,
                                    int,
                                    int,
                                    cudaStream_t);

__global__ void d_gelu_func(float* d_output,
                            const float* gelu_input,
                            const float* bias,
                            int row_stride,
                            int iterations)
{
    int row = blockIdx.x;
    int id = threadIdx.x;
    int loop_stride = blockDim.x;

    float4* d_output_cast = reinterpret_cast<float4*>(d_output);
    const float4* gelu_input_cast = reinterpret_cast<const float4*>(gelu_input);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float4 output_data = d_output_cast[row * row_stride + i * loop_stride + id];
            float4 gelu_input_data = gelu_input_cast[row * row_stride + i * loop_stride + id];
            float4 bias_data = bias_cast[i * loop_stride + id];

            gelu_input_data.x += bias_data.x;
            gelu_input_data.y += bias_data.y;
            gelu_input_data.z += bias_data.z;
            gelu_input_data.w += bias_data.w;

            output_data.x *= d_gelu(gelu_input_data.x);
            output_data.y *= d_gelu(gelu_input_data.y);
            output_data.z *= d_gelu(gelu_input_data.z);
            output_data.w *= d_gelu(gelu_input_data.w);

            d_output_cast[row * row_stride + i * loop_stride + id] = output_data;
        }
    }
}

__global__ void d_gelu_func(__half* d_output,
                            const __half* gelu_input,
                            const __half* bias,
                            int row_stride,
                            int iterations)
{
#if __CUDA_ARCH__ >= 700
    int row = blockIdx.x;
    int id = threadIdx.x;
    int loop_stride = blockDim.x;

    float2* d_output_cast = reinterpret_cast<float2*>(d_output);
    const float2* gelu_input_cast = reinterpret_cast<const float2*>(gelu_input);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        if (i * loop_stride + id < row_stride) {
            float2 output_data = d_output_cast[row * row_stride + i * loop_stride + id];
            float2 gelu_input_data = gelu_input_cast[row * row_stride + i * loop_stride + id];
            float2 bias_vec = bias_cast[i * loop_stride + id];

            __half2* output_data_half = reinterpret_cast<__half2*>(&output_data);
            __half2* gelu_input_data_half = reinterpret_cast<__half2*>(&gelu_input_data);
            __half2* bias_half = reinterpret_cast<__half2*>(&bias_vec);

            float2 output_half_0 = __half22float2(output_data_half[0]);
            float2 output_half_1 = __half22float2(output_data_half[1]);

            float2 gelu_input_half_0 = __half22float2(gelu_input_data_half[0]);
            float2 gelu_input_half_1 = __half22float2(gelu_input_data_half[1]);

            float2 bias_half_0 = __half22float2(bias_half[0]);
            float2 bias_half_1 = __half22float2(bias_half[1]);

            gelu_input_half_0.x += bias_half_0.x;
            gelu_input_half_0.y += bias_half_0.y;
            gelu_input_half_1.x += bias_half_1.x;
            gelu_input_half_1.y += bias_half_1.y;

            output_half_0.x *= d_gelu(gelu_input_half_0.x);
            output_half_0.y *= d_gelu(gelu_input_half_0.y);
            output_half_1.x *= d_gelu(gelu_input_half_1.x);
            output_half_1.y *= d_gelu(gelu_input_half_1.y);

            float2 result;
            __half2* result_half2 = reinterpret_cast<__half2*>(&result);

            result_half2[0] = __float22half2_rn(output_half_0);
            result_half2[1] = __float22half2_rn(output_half_1);

            d_output_cast[row * row_stride + i * loop_stride + id] = result;
        }
    }
#endif
}

template <typename T>
void launch_d_gelu(T* d_output,
                   const T* input,
                   const T* bias,
                   int intermediate_size,
                   int batch_size,
                   cudaStream_t stream)
{
    int iterations = (intermediate_size + 1023) / 1024;
    int threads = (intermediate_size - 1) / (iterations * 4) + 1;
    dim3 block_dims(threads);
    dim3 grid_dims(batch_size);

    d_gelu_func<<<grid_dims, block_dims, 0, stream>>>(d_output, input, bias, intermediate_size / 4, iterations);
}

template void launch_d_gelu<float>(float*, const float*, const float*, int, int, cudaStream_t);
template void launch_d_gelu<__half>(__half*, const __half*, const __half*, int, int, cudaStream_t);

template <typename T>
class Gelu {
public:
    struct Config {
        uint32_t intermediate_size;
        Config(uint32_t inter_size) : intermediate_size(inter_size) {}
    };

    Gelu(const Config& config) : _config(config) {}

    virtual ~Gelu() {}

    void ForwardWithBiasAdd(int bsz, 
                            Buffer <T>* input_buf,
                            Buffer <T>* bias,
                            Buffer <T>* output,
                            ScheduleEngine* SE,
                            int q_index=0)
    {

        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        input_buf->copyH2D(SE->compute);
        bias->copyH2D(SE->compute);
        output->copyH2D(SE->compute);  

        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time:%lf\n",sw.GetTimeInSeconds());
            sw.restart();
        #endif

        launch_bias_gelu<T>(input_buf->get_device_data(), 
                            bias->get_device_data(), 
                            output->get_device_data(), 
                            _config.intermediate_size, 
                            bsz, 
                            SE->getStream(q_index));
    
        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time:%lf\n",sw.GetTimeInSeconds());
            sw.restart();
        #endif

        output->copyD2H(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time:%lf\n",sw.GetTimeInSeconds());
        #endif        
    }

    void Backward(int bsz,
                Buffer <T>* d_output,
                Buffer <T>* input_buf,
                Buffer <T>* bias,
                ScheduleEngine* SE)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        d_output->copyH2D(SE->compute);
        input_buf->copyH2D(SE->compute);
        bias->copyH2D(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif

        launch_d_gelu<T>(d_output->get_device_data(),
                        input_buf->get_device_data(),
                        bias->get_device_data(),
                        _config.intermediate_size,
                        bsz,
                        SE->getStream(0));

        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif                        
                        
        d_output->copyD2H(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time: %lf\n", sw.GetTimeInSeconds());
        #endif               
    }

    void BackwardFineGrained(int bsz,
                int nq,
                Buffer <T>* d_output,
                Buffer <T>* input_buf,
                Buffer <T>* bias,
                ScheduleEngine* SE)
    {
        input_buf->copyH2D(SE->compute);
        bias->copyH2D(SE->compute);

        int offset = 0;
        int offset_size = bsz * _config.intermediate_size / nq;

        #if DEBUG
            std::cout << "offset_size=" << offset_size << std::endl;
            std::cout << "input volume=" << bsz*config_.inputSize << std::endl;
            std::cout << "output volume=" << 3*bsz*config_.inputSize << std::endl;
        #endif

    
        for (int i = 0; i < nq; i++)
        {
            offset = i * offset_size;   
            d_output->copyH2D(SE->compute, offset, nq, i);
            // out->copyH2D(SE->compute, offset, nq, i);

            #if DEBUG
                std::cout << "\x1b[31;1mqueue index=" << i << "\x1b[0m" << std::endl;
                std::cout << "input offset=" << offset << std::endl;
                std::cout << "output offset=" << 3*offset << std::endl;
            #endif

            cublasSetStream(SE->handle, SE->compute[i]);        
            
            launch_d_gelu<T>(d_output->get_device_data(offset),
                        input_buf->get_device_data(),
                        bias->get_device_data(),
                        _config.intermediate_size,
                        bsz,
                        SE->getStream(0));
            
            d_output->copyD2H(SE->compute, offset, nq, i);
        }
    }

    private:
        Config _config;
};