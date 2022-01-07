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

template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(float* vals,
                             const float* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = (seq_length > tbSeq) ? seq_length : tbSeq;
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;

    float4* val_cast = reinterpret_cast<float4*>(vals);
    const float4* attn_mask_cast = reinterpret_cast<const float4*>(attn_mask);

    float4 data[MAX_THREAD_ITERATIONS];

    float infinity = FLT_MAX;
    float max_val = -infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float4 mask = attn_mask_cast[mask_offset + data_id];
            data[i] = val_cast[data_offset + data_id];

            data[i].x += mask.x;
            data[i].y += mask.y;
            data[i].z += mask.z;
            data[i].w += mask.w;

            max_val = (data[i].x > max_val ? data[i].x : max_val);
            max_val = (data[i].y > max_val ? data[i].y : max_val);
            max_val = (data[i].z > max_val ? data[i].z : max_val);
            max_val = (data[i].w > max_val ? data[i].w : max_val);
        } else {
            data[i].x = -infinity;
            data[i].y = -infinity;
            data[i].z = -infinity;
            data[i].w = -infinity;
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        b.sync();

        if (lane < warp_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shfl(max_val, threadIdx.x / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        data[i].x = __expf(data[i].x - max_val);
        data[i].y = __expf(data[i].y - max_val);
        data[i].z = __expf(data[i].z - max_val);
        data[i].w = __expf(data[i].w - max_val);

        sum += (data[i].x + data[i].y + data[i].z + data[i].w);
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        b.sync();

        if (lane < warp_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

        sum = g.shfl(sum, threadIdx.x / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        data[i].x /= sum;
        data[i].y /= sum;
        data[i].z /= sum;
        data[i].w /= sum;

        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) val_cast[data_offset + data_id] = data[i];
    }
}

template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(__half* vals,
                             const __half* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
#if __CUDA_ARCH__ >= 700
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = (seq_length > tbSeq) ? seq_length : tbSeq;
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;

    float2* val_cast = reinterpret_cast<float2*>(vals);
    const float2* attn_mask_cast = reinterpret_cast<const float2*>(attn_mask);

    val_cast += data_offset;
    attn_mask_cast += mask_offset;

    float2 low_data[MAX_THREAD_ITERATIONS];
    float2 high_data[MAX_THREAD_ITERATIONS];

    float infinity = FLT_MAX;
    float max_val = -infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float2 data = val_cast[data_id];
            float2 mask = attn_mask_cast[data_id];

            __half2* data_arr = reinterpret_cast<__half2*>(&data);
            __half2* mask_arr = reinterpret_cast<__half2*>(&mask);

            low_data[i] = __half22float2(data_arr[0]);
            high_data[i] = __half22float2(data_arr[1]);
            float2 low_mask = __half22float2(mask_arr[0]);
            float2 high_mask = __half22float2(mask_arr[1]);

            low_data[i].x += low_mask.x;
            low_data[i].y += low_mask.y;
            high_data[i].x += high_mask.x;
            high_data[i].y += high_mask.y;

            max_val = (low_data[i].x > max_val ? low_data[i].x : max_val);
            max_val = (low_data[i].y > max_val ? low_data[i].y : max_val);
            max_val = (high_data[i].x > max_val ? high_data[i].x : max_val);
            max_val = (high_data[i].y > max_val ? high_data[i].y : max_val);
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        b.sync();

        if (lane < warp_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shfl(max_val, threadIdx.x / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            low_data[i].x = __expf(low_data[i].x - max_val);
            low_data[i].y = __expf(low_data[i].y - max_val);
            high_data[i].x = __expf(high_data[i].x - max_val);
            high_data[i].y = __expf(high_data[i].y - max_val);

            sum += (low_data[i].x + low_data[i].y + high_data[i].x + high_data[i].y);
        }
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        b.sync();

        if (lane < warp_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride)
            iters = warp_num / (iteration_stride / max_threads_in_sequence);

        for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

        sum = g.shfl(sum, threadIdx.x / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float2 result_f;
            __half2* result_h = reinterpret_cast<__half2*>(&result_f);

            low_data[i].x /= sum;
            low_data[i].y /= sum;
            high_data[i].x /= sum;
            high_data[i].y /= sum;

            result_h[0] = __float22half2_rn(low_data[i]);
            result_h[1] = __float22half2_rn(high_data[i]);

            val_cast[data_id] = result_f;
        }
    }

#endif
}

template <typename T>
void launch_attn_softmax(T*, const T*, int, int, int, cudaStream_t);

template <>
void launch_attn_softmax<float>(float* vals,
                                const float* attn_mask,
                                int batch_size,
                                int heads,
                                int sequence_length,
                                cudaStream_t stream)
{
    const int threads = 128;
    int seq_length4 = sequence_length / 4;

    int block_compute_size =
        (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4)))) : 1);
    dim3 grid_dim(batch_size, heads * sequence_length / block_compute_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                            subblock_max_workload * threads)
                                         : threads);
    int iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

    if (sequence_length <= 8)
        attn_softmax<2, (threads / 2), 2>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 16)
        attn_softmax<4, (threads / 4), 4>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 32)
        attn_softmax<8, (threads / 8), 8>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 64)
        attn_softmax<16, (threads / 16), 16>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 128)
        attn_softmax<32, (threads / 32), 32>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 256)
        attn_softmax<32, (threads / 64), 64>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else {
        const int threads = 256;
        block_compute_size =
            (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4))))
                                   : 1);
        dim3 grid_dim(batch_size, heads * sequence_length / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);
        iterations =
            (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                     : MAX_THREAD_ITERATIONS);
        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else
            throw std::runtime_error(
                "Unsupport Seq_Length! Check the restriction of the max_threads and "
                "max_thread_iterations!");
    }
}

template <>
void launch_attn_softmax<__half>(__half* vals,
                                 const __half* attn_mask,
                                 int batch_size,
                                 int heads,
                                 int sequence_length,
                                 cudaStream_t stream)
{
    const int threads = 128;
    int seq_length4 = sequence_length / 4;

    int block_compute_size =
        (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4)))) : 1);
    dim3 grid_dim(batch_size, heads * sequence_length / block_compute_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                            subblock_max_workload * threads)
                                         : threads);

    int iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

    if (sequence_length <= 8)
        attn_softmax<2, (threads / 2), 2>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 16)
        attn_softmax<4, (threads / 4), 4>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 32)
        attn_softmax<8, (threads / 8), 8>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 64)
        attn_softmax<16, (threads / 16), 16>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 128)
        attn_softmax<32, (threads / 32), 32>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 256)
        attn_softmax<32, (threads / 64), 64>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else {
        const int threads = 256;
        block_compute_size =
            (seq_length4 < threads ? (int)pow(2.0, floor(log2((float)(threads / seq_length4))))
                                   : 1);
        dim3 grid_dim(batch_size, heads * sequence_length / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);
        iterations =
            (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                     : MAX_THREAD_ITERATIONS);
        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else
            throw std::runtime_error(
                "Unsupport Seq_Length! Check the restriction of the max_threads and "
                "max_thread_iterations!");
    }
}

template <typename T, int ITERATIONS>
__global__ void softmax_backward_kernel_v2(T* grad /* input & output*/,
                                           const T* output,
                                           int softmax_length)
{
    int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int offset = batch_idx * softmax_length + threadIdx.x;

    grad += offset;
    output += offset;

    T grad_reg[ITERATIONS];
    T output_reg[ITERATIONS];
    float sum = 0.0;

#pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int curr_idx = threadIdx.x + i * WARP_SIZE;
        if (curr_idx < softmax_length) {
            grad_reg[i] = grad[i * WARP_SIZE];
            output_reg[i] = output[i * WARP_SIZE];
            sum += (float)grad_reg[i] * (float)output_reg[i];
        }
    }

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_xor(sum, i);

#pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int curr_idx = threadIdx.x + i * WARP_SIZE;
        if (curr_idx < softmax_length)
            grad[i * WARP_SIZE] = (float)output_reg[i] * ((float)grad_reg[i] - sum);
    }
}

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     cudaStream_t stream);

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     cudaStream_t stream)
{
    const int warps_per_block = 4;
    dim3 grid_dim(batch_size * heads * seq_length / warps_per_block);
    dim3 block_dim(WARP_SIZE, warps_per_block);

    if (seq_length <= 32)
        softmax_backward_kernel_v2<T, 1>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 64)
        softmax_backward_kernel_v2<T, 2>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 128)
        softmax_backward_kernel_v2<T, 4>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 256)
        softmax_backward_kernel_v2<T, 8>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 384)
        softmax_backward_kernel_v2<T, 12>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 512)
        softmax_backward_kernel_v2<T, 16>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 768)
        softmax_backward_kernel_v2<T, 24>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 1024)
        softmax_backward_kernel_v2<T, 32>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else if (seq_length <= 2048)
        softmax_backward_kernel_v2<T, 64>
            <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
    else
        throw std::runtime_error(
            std::string("Special sequence length found in softmax backward, seq_length: ") +
            std::to_string(seq_length));
}



template <typename T>
class Softmax {
public:
    struct Config {
        size_t batchSize;
        size_t heads;
        size_t seq_length;
        size_t prob_depth;
        float temprature;
        bool mem_alloc;
        Config(size_t batch, size_t h, size_t seq, int prob_size = 0, bool mem_alloc = false)
            : batchSize(batch),
              heads(h),
              seq_length(seq),
              prob_depth(prob_size),
              temprature(1.0),
              mem_alloc(mem_alloc)
        {
        }
    };

    Softmax(Config config) : config_(config) {}

    ~Softmax() {}

    void ForwardCheckpoint(int bsz, Buffer <T>* vals, Buffer <T>* attn_mask, ScheduleEngine* SE, int q_index=0)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif
	
        vals->copyH2D(SE->compute);
        attn_mask->copyH2D(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time:%lf\n",sw.GetTimeInSeconds());
            sw.restart();
        #endif
        
        launch_attn_softmax<T>(vals->get_device_data(), 
                               attn_mask->get_device_data(), 
                               bsz, 
                               config_.heads, 
                               config_.seq_length, 
                               SE->getStream(q_index));

        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time:%lf\n",sw.GetTimeInSeconds());
            sw.restart();
        #endif
        
        vals->copyD2H(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time:%lf\n",sw.GetTimeInSeconds());
            sw.restart();
        #endif
    }

    void Backward(int bsz,
                Buffer<T>* out_grad,
                Buffer<T>* soft_out,
                ScheduleEngine * SE)
    {
        launch_attn_softmax_backward_v2<T>(out_grad->get_device_data(),
                                        soft_out->get_device_data(),
                                        bsz,
                                        config_.heads,
                                        config_.seq_length,
                                        SE->getStream(0));
    }




    inline size_t GetProbDepth() const { return config_.prob_depth; }

    inline size_t GetBatchSize() const { return config_.batchSize; }

    inline size_t GetNumHeads() const { return config_.heads; }

    inline size_t GetSeqLength() const { return config_.seq_length; }

    inline void SetSeqLength(size_t seq_len) { config_.seq_length = seq_len; }

    private:
        Config config_;
};