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

__global__ void dropout_kernel(const int N,
                               const float ratio,
                               float* out,
                               const float* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    CUDA_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float4 rand = curand_uniform4(&state);
        uint8_t m[unroll_factor];

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        int i = j * unroll_factor;

        mask[i] = (uint8_t)m[0];
        mask[i + 1] = (uint8_t)m[1];
        mask[i + 2] = (uint8_t)m[2];
        mask[i + 3] = (uint8_t)m[3];

        out[i] = Xdata[i] * scale * m[0];
        out[i + 1] = Xdata[i + 1] * scale * m[1];
        out[i + 2] = Xdata[i + 2] * scale * m[2];
        out[i + 3] = Xdata[i + 3] * scale * m[3];
    }
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        float4 rand = curand_uniform4(&state);
        float* rand_data = &(rand.x);
        int k = 0;
        for (int i = high_index; i < N; i++) {
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            out[i] = Xdata[i] * scale * m;
            mask[i] = m;
        }
    }
}

__global__ void dropout_kernel(const int N,
                               const float ratio,
                               __half* out,
                               const __half* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

#ifdef __STOCHASTIC_MODE__

    const __half2 h_scale = __float2half2_rn(scale);
    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    uint32_t m_32;
    uint8_t* m = reinterpret_cast<uint8_t*>(&m_32);

    float2 result_f;
    __half2* result_h = reinterpret_cast<__half2*>(&result_f);
    __half2 mask_h[2];
    float2 mask_f[2];

    CUDA_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float2 x_f = x_cast[j];
        __half2* x_h = reinterpret_cast<__half2*>(&x_f);

        float4 rand = curand_uniform4(&state);

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) mask_f_data[i] = (float)(m[i]);

        mask_h[0] = __float22half2_rn(mask_f[0]);
        mask_h[1] = __float22half2_rn(mask_f[1]);

        result_h[0] = x_h[0] * h_scale * mask_h[0];
        result_h[1] = x_h[1] * h_scale * mask_h[1];

        out_cast[j] = result_f;

        mask_cast[j] = m_32;
    }

#else

    CUDA_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        const __half2* vals_half = reinterpret_cast<const __half2*>(Xdata + i);
        float2 vals_half_f[2];
        vals_half_f[0] = __half22float2(vals_half[0]);
        vals_half_f[1] = __half22float2(vals_half[1]);

        uint8_t m[unroll_factor];
        float4 rand = curand_uniform4(&state);
        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        out[i] = __float2half(vals_half_f[0].x * scale * m[0]);
        out[i + 1] = __float2half(vals_half_f[0].y * scale * m[1]);
        out[i + 2] = __float2half(vals_half_f[1].x * scale * m[2]);
        out[i + 3] = __float2half(vals_half_f[1].y * scale * m[3]);

        mask[i] = m[0];
        mask[i + 1] = m[1];
        mask[i + 2] = m[2];
        mask[i + 3] = m[3];
    }

#endif
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        float4 rand = curand_uniform4(&state);
        float* rand_data = &(rand.x);
        int k = 0;
        for (int i = high_index; i < N; i++) {
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            out[i] = __float2half((float)Xdata[i] * scale * m);
            mask[i] = m;
        }
    }
}

__global__ void dropout_kernel_bwd(const int N,
                                   const float ratio,
                                   const float* Xdata,
                                   float* out,
                                   uint8_t* mask,
                                   std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    CUDA_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        out[i] = mask[i] ? Xdata[i] * scale : 0.0;
        out[i + 1] = mask[i + 1] ? Xdata[i + 1] * scale : 0.0;
        out[i + 2] = mask[i + 2] ? Xdata[i + 2] * scale : 0.0;
        out[i + 3] = mask[i + 3] ? Xdata[i + 3] * scale : 0.0;
    }
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        for (int i = high_index; i < N; i++) { out[i] = mask[i] ? Xdata[i] * scale : 0.0; }
    }
}

__global__ void dropout_kernel_bwd(const int N,
                                   const float ratio,
                                   const __half* Xdata,
                                   __half* out,
                                   uint8_t* mask,
                                   std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);

#ifdef __STOCHASTIC_MODE__

    const __half2 h_scale = __float2half2_rn(scale);

    const float2* x_cast = reinterpret_cast<const float2*>(Xdata);
    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_cast = reinterpret_cast<uint32_t*>(mask);

    CUDA_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        float2 x_f = x_cast[j];
        __half2* x_h = reinterpret_cast<__half2*>(&x_f);

        uint32_t m_32 = mask_cast[j];
        uint8_t* m = (uint8_t*)&m_32;

        __half2 mask_h[2];
        float2 mask_f[2];

        float* mask_f_data = &mask_f[0].x;
#pragma unroll
        for (int i = 0; i < unroll_factor; i++) mask_f_data[i] = (float)(m[i]);

#pragma unroll
        for (int i = 0; i < 2; i++) mask_h[i] = __float22half2_rn(mask_f[i]);

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = x_h[0] * h_scale * mask_h[0];
        result_h[1] = x_h[1] * h_scale * mask_h[1];

        out_cast[j] = result_f;
    }

#else

    const __half h_scale = __float2half(scale);
    const __half h_zero = __float2half(0.0);

    CUDA_1D_KERNEL_LOOP(j, N / unroll_factor)
    {
        int i = j * unroll_factor;

        const __half2* vals_half = reinterpret_cast<const __half2*>(Xdata + i);

        uint8_t* m = mask + i;

        float2 vals_half_f[2];

        vals_half_f[0] = __half22float2(vals_half[0]);
        vals_half_f[1] = __half22float2(vals_half[1]);

        out[i] = __float2half(vals_half_f[0].x * scale * m[0]);
        out[i + 1] = __float2half(vals_half_f[0].y * scale * m[1]);
        out[i + 2] = __float2half(vals_half_f[1].x * scale * m[2]);
        out[i + 3] = __float2half(vals_half_f[1].y * scale * m[3]);
    }

#endif
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        for (int i = high_index; i < N; i++) {
            out[i] = __float2half((float)Xdata[i] * scale * mask[i]);
        }
    }
}

__global__ void dropout_kernel(const int N,
                               const int dim,
                               const float ratio,
                               const float* bias,
                               float* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x % (dim / unroll_factor);

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float4* Xdata_cast = reinterpret_cast<float4*>(Xdata);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);
    const float4* bias_cast = reinterpret_cast<const float4*>(bias);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = curand_uniform4(&state);
        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        float4 x_data = Xdata_cast[j];
        float4 b_data = bias_cast[tid];

        x_data.x += b_data.x;
        x_data.y += b_data.y;
        x_data.z += b_data.z;
        x_data.w += b_data.w;

        x_data.x = x_data.x * scale * m[0];
        x_data.y = x_data.y * scale * m[1];
        x_data.z = x_data.z * scale * m[2];
        x_data.w = x_data.w * scale * m[3];

        mask_32[j] = m_32;
        Xdata_cast[j] = x_data;
    }
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        float4 rand = curand_uniform4(&state);
        float* rand_data = &(rand.x);
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = Xdata[i] + bias[threadIdx.x % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            Xdata[i] = x_data * scale * m;
            mask[i] = m;
        }
    }
}

__global__ void dropout_kernel(const int N,
                               const int dim,
                               const float ratio,
                               const __half* bias,
                               __half* Xdata,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x % (dim / unroll_factor);

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float2* Xdata_cast = reinterpret_cast<float2*>(Xdata);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = curand_uniform4(&state);

        float2 data_f;
        __half2* data_h = reinterpret_cast<__half2*>(&data_f);

        float2 bias_f;
        __half2* bias_h = reinterpret_cast<__half2*>(&bias_f);

        data_f = Xdata_cast[j];
        bias_f = bias_cast[tid];

        float2 data_h_0 = __half22float2(data_h[0]);
        float2 data_h_1 = __half22float2(data_h[1]);

        float2 bias_h_0 = __half22float2(bias_h[0]);
        float2 bias_h_1 = __half22float2(bias_h[1]);

        data_h_0.x += bias_h_0.x;
        data_h_0.y += bias_h_0.y;
        data_h_1.x += bias_h_1.x;
        data_h_1.y += bias_h_1.y;

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        data_h_0.x = __float2half(data_h_0.x * scale * m[0]);
        data_h_0.y = __float2half(data_h_0.y * scale * m[1]);
        data_h_1.x = __float2half(data_h_1.x * scale * m[2]);
        data_h_1.y = __float2half(data_h_1.y * scale * m[3]);

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = __float22half2_rn(data_h_0);
        result_h[1] = __float22half2_rn(data_h_1);

        Xdata_cast[j] = result_f;
        mask_32[j] = m_32;
    }
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        float4 rand = curand_uniform4(&state);
        float* rand_data = &(rand.x);
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = (float)Xdata[i] + (float)bias[threadIdx.x % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            Xdata[i] = __float2half(x_data * scale * m);
            mask[i] = m;
        }
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* vals,
                    uint8_t* mask,
                    int total_count,
                    int dim,
                    float ratio,
                    cudaStream_t stream,
                    bool bwd)
{
    assert(unroll_factor == 4);

    dim3 grid_dim = DS_GET_BLOCKS(total_count / unroll_factor);
    dim3 block_dim = DS_CUDA_NUM_THREADS;

    if (dim > 512) {
        block_dim.x >>= 1;
        grid_dim.x <<= 1;
    }
    uint64_t inc = total_count / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = getSeed(inc, 42);
    if (bwd)
        dropout_kernel_bwd<<<grid_dim, block_dim, 0, stream>>>(
            total_count, ratio, vals, out, mask, seed);
    else
        dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(
            total_count, ratio, out, vals, mask, seed);
}

template void launch_dropout(float* out,
                             const float* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             cudaStream_t stream,
                             bool);
template void launch_dropout(__half* out,
                             const __half* vals,
                             uint8_t* mask,
                             int total_count,
                             int dim,
                             float ratio,
                             cudaStream_t stream,
                             bool);

template <typename T>
void launch_dropout(T* out,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    cudaStream_t stream)
{
    assert(unroll_factor == 4);

    int total_count = batch * dim / unroll_factor;

    dim3 grid_dim = DS_GET_BLOCKS(total_count);
    dim3 block_dim = DS_CUDA_NUM_THREADS;

    uint64_t inc = (batch * dim) / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = getSeed(inc, 42);

    dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(
        total_count, dim, ratio, bias, out, mask, seed);
}

template void launch_dropout(float*,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);
template void launch_dropout(__half*,
                             const __half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);

__global__ void dropout_kernel(const int N,
                               const int dim,
                               const float ratio,
                               const float* input,
                               const float* residual,
                               const float* bias,
                               float* out,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x % (dim / unroll_factor);

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float4* out_cast = reinterpret_cast<float4*>(out);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);

    const float4* bias_cast = reinterpret_cast<const float4*>(bias);
    const float4* residual_cast = reinterpret_cast<const float4*>(residual);
    const float4* input_cast = reinterpret_cast<const float4*>(input);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = curand_uniform4(&state);

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        float4 out_data;
        float4 b_data = bias_cast[tid];
        float4 res_data = residual_cast[j];
        float4 inp_data = input_cast[j];

        out_data.x = (b_data.x + inp_data.x);
        out_data.y = (b_data.y + inp_data.y);
        out_data.z = (b_data.z + inp_data.z);
        out_data.w = (b_data.w + inp_data.w);

        out_data.x = out_data.x * scale * m[0];
        out_data.y = out_data.y * scale * m[1];
        out_data.z = out_data.z * scale * m[2];
        out_data.w = out_data.w * scale * m[3];

        out_data.x += res_data.x;
        out_data.y += res_data.y;
        out_data.z += res_data.z;
        out_data.w += res_data.w;

        mask_32[j] = m_32;
        out_cast[j] = out_data;
    }
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        float4 rand = curand_uniform4(&state);
        float* rand_data = &(rand.x);
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = input[i] + bias[threadIdx.x % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            x_data = x_data * scale * m;
            x_data += residual[i];

            out[i] = x_data;
            mask[i] = m;
        }
    }
}

__global__ void dropout_kernel(const int N,
                               const int dim,
                               const float ratio,
                               const __half* input,
                               const __half* residual,
                               const __half* bias,
                               __half* out,
                               uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed)
{
    const float scale = 1. / (1. - ratio);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x % (dim / unroll_factor);

    curandStatePhilox4_32_10_t state;
    curand_init(seed.first, idx, seed.second, &state);

    float2* out_cast = reinterpret_cast<float2*>(out);
    uint32_t* mask_32 = reinterpret_cast<uint32_t*>(mask);

    const float2* bias_cast = reinterpret_cast<const float2*>(bias);
    const float2* residual_cast = reinterpret_cast<const float2*>(residual);
    const float2* input_cast = reinterpret_cast<const float2*>(input);

    CUDA_1D_KERNEL_LOOP(j, N)
    {
        float4 rand = curand_uniform4(&state);

        float2 data_f;
        __half2* data_h = reinterpret_cast<__half2*>(&data_f);

        float2 bias_f;
        __half2* bias_h = reinterpret_cast<__half2*>(&bias_f);

        float2 residual_f;
        __half2* residual_h = reinterpret_cast<__half2*>(&residual_f);

        float2 input_f;
        __half2* input_h = reinterpret_cast<__half2*>(&input_f);

        bias_f = bias_cast[tid];
        residual_f = residual_cast[j];
        input_f = input_cast[j];

        float2 data_h_0 = __half22float2(data_h[0]);
        float2 data_h_1 = __half22float2(data_h[1]);

        float2 bias_h_0 = __half22float2(bias_h[0]);
        float2 bias_h_1 = __half22float2(bias_h[1]);

        float2 residual_h_0 = __half22float2(residual_h[0]);
        float2 residual_h_1 = __half22float2(residual_h[1]);

        float2 input_h_0 = __half22float2(input_h[0]);
        float2 input_h_1 = __half22float2(input_h[1]);

        data_h_0.x = (bias_h_0.x + input_h_0.x);
        data_h_0.y = (bias_h_0.y + input_h_0.y);
        data_h_1.x = (bias_h_1.x + input_h_1.x);
        data_h_1.y = (bias_h_1.y + input_h_1.y);

        uint32_t m_32;
        uint8_t* m = (uint8_t*)&m_32;

        m[0] = (uint8_t)(rand.x > ratio);
        m[1] = (uint8_t)(rand.y > ratio);
        m[2] = (uint8_t)(rand.z > ratio);
        m[3] = (uint8_t)(rand.w > ratio);

        data_h_0.x = __float2half(data_h_0.x * scale * m[0]);
        data_h_0.y = __float2half(data_h_0.y * scale * m[1]);
        data_h_1.x = __float2half(data_h_1.x * scale * m[2]);
        data_h_1.y = __float2half(data_h_1.y * scale * m[3]);

        data_h_0.x += residual_h_0.x;
        data_h_0.y += residual_h_0.y;
        data_h_1.x += residual_h_1.x;
        data_h_1.y += residual_h_1.y;

        float2 result_f;
        __half2* result_h = reinterpret_cast<__half2*>(&result_f);

        result_h[0] = __float22half2_rn(data_h_0);
        result_h[1] = __float22half2_rn(data_h_1);

        out_cast[j] = result_f;
        mask_32[j] = m_32;
    }
    int high_index =
        ((((N / unroll_factor) - 1) / blockDim.x + 1) * (unroll_factor * blockDim.x)) + threadIdx.x;
    if (N > high_index) {
        float4 rand = curand_uniform4(&state);
        float* rand_data = &(rand.x);
        int k = 0;
        for (int i = high_index; i < N; i++) {
            float x_data = (float)input[i] + (float)bias[threadIdx.x % dim];
            uint8_t m = (uint8_t)(rand_data[k++] > ratio);
            x_data = x_data * scale * m;
            x_data += (float)residual[i];

            out[i] = __float2half(x_data);
            mask[i] = m;
        }
    }
}

template <typename T>
void launch_dropout(T* out,
                    const T* input,
                    const T* residual,
                    const T* bias,
                    uint8_t* mask,
                    int batch,
                    int dim,
                    float ratio,
                    cudaStream_t stream)
{
    assert(unroll_factor == 4);

    int total_count = batch * dim / unroll_factor;
    dim3 grid_dim = DS_GET_BLOCKS(total_count);
    dim3 block_dim = DS_CUDA_NUM_THREADS;

    uint64_t inc = (batch * dim) / grid_dim.x / block_dim.x;
    std::pair<uint64_t, uint64_t> seed = getSeed(inc, 42);

    dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(
        total_count, dim, ratio, input, residual, bias, out, mask, seed);
}

template void launch_dropout(float*,
                             const float*,
                             const float* residual,
                             const float* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);
template void launch_dropout(__half*,
                             const __half*,
                             const __half* residual,
                             const __half* bias,
                             uint8_t* mask,
                             int batch,
                             int dim,
                             float ratio,
                             cudaStream_t stream);


template <typename T>
class Dropout {
public:
    struct Config {
        float ratio;
        uint32_t dim;
        bool training;

        Config(float r, uint32_t d) : ratio(r), dim(d), training(true) {}

        float RATIO() const { return training ? ratio : 0.0; }
        inline void SetDim(uint32_t d) { dim = d; }
    };

    Dropout(const Config& config) : _config(config), _mask(nullptr) {}

    virtual ~Dropout() {}

    void Forward(int bsz, 
                 Buffer<T>* out, 
                 Buffer<T>* vals, 
                 ScheduleEngine* SE, 
                 bool bwd = false,
                 int q_index = 0)
    {
        if ( _mask == nullptr ) 
        {
            std::cout << "Need to invoke SetMask, as dropout _mask is currently set to nullptr" << std::endl;
            exit(EXIT_FAILURE);
        }
#if EVENT_PROFILE
        Stopwatch sw;
        sw.restart();
#endif
        launch_dropout<T>(
            out->get_device_data(), vals->get_device_data(), _mask, bsz * _config.dim, _config.dim, _config.RATIO(), SE->getStream(q_index), bwd);
#if EVENT_PROFILE
        sw.stop();
        printf("Kernel Time:%lf\n",sw.GetTimeInSeconds());
        sw.restart();
#endif    
}

    /* void ForwardWithBias(int bsz, T* vals, const T* bias, cudaStream_t stream)
    {
        launch_dropout<T>(vals, bias, _mask, bsz, _config.dim, _config.RATIO(), stream);
    } */

    void ForwardWithBias(int bsz,
                         Buffer<T>* out,
                         Buffer<T>* vals,
                         Buffer<T>* residual,
                         Buffer<T>* bias,
                         ScheduleEngine* SE,
                         int q_index=0)
    {
#if EVENT_PROFILE
        Stopwatch sw;
        sw.restart();
#endif      

	    launch_dropout<T>(
            out->get_device_data(), 
            vals->get_device_data(), 
            residual->get_device_data(), 
            bias->get_device_data(), 
            _mask, 
            bsz, 
            _config.dim, 
            _config.RATIO(), 
            SE->getStream(q_index));
#if EVENT_PROFILE
        sw.stop();
        printf("Kernel Time:%lf\n",sw.GetTimeInSeconds());
        sw.restart();
#endif    
}

    bool HasDropout() const { return _config.RATIO() > 0.0; }

    void SetTrainingMode(bool training) { _config.training = training; }

    void SetMask(Buffer<uint8_t>* mask)
    {
        if (!mask) { throw std::runtime_error("Dropout mask is null."); }

        _mask = mask->get_device_data();
    }

    Config GetConfig() const { return _config; }

    inline void SetDimension(uint32_t dim) { _config.SetDim(dim); }

private:
    uint8_t* _mask;
    Config _config;
};
