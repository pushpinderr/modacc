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

template void launch_bias_gelu<float>(const float*, const float*, float*, int, int, cudaStream_t);
template void launch_bias_gelu<__half>(const __half*, const __half*, __half*, int, int, cudaStream_t);


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
                            ScheduleEngine* SE)
    {
        launch_bias_gelu<T>(input_buf->get_device_data(), 
                            bias->get_device_data(), 
                            output->get_device_data(), 
                            _config.intermediate_size, 
                            bsz, 
                            SE->compute[0]);
    }

private:
    Config _config;
};


// int main(int argc, char *argv[])
// {
//     // PLEASE NOTE: number of queues must be less than batch_size
//     int batch_size = atoi(argv[1]);
//     int sequence_length = atoi(argv[2]);
//     int intermediate_size = atoi(argv[3]);
//     int nq = atoi(argv[4]);
//     int bsz = sequence_length * batch_size;

//     std::cout << "batch size=" << batch_size << std::endl;
//     std::cout << "sequence length=" << sequence_length << std::endl;
//     std::cout << "intermediate size=" << intermediate_size << std::endl;
//     std::cout << "number of queues=" << nq << std::endl;
 
//     ScheduleEngine SE(nq);
//     Buffer<float> input(bsz * sequence_length, &SE);
//     Buffer<float> bias(batch_size * sequence_length, &SE);
//     Buffer<float> output(batch_size * sequence_length, &SE);
//     // Gelu<float> GeLU(Gelu<float>::Config(intermediate_size));
//     Gelu<float> GeLU{Gelu<float>::Config(intermediate_size)};

//     for ( int i = 0; i < nq; i++ ) 
//     {
        
//     }
//     GeLU.ForwardWithBiasAdd(bsz, &input, &bias, &output, &SE);
//     CHECK(cudaThreadSynchronize());
//     printf("Executed Gelu\n");
// }