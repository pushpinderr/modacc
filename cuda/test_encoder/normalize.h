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

__global__ void fused_bias_residual_layer_norm(float* vals,
                                               const float* residual,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               float* vars,
                                               float* means,
                                               int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id / WARP_SIZE;

    float vals_arr[NORM_REG];
    __shared__ float shr[MAX_WARP_NUM];

    residual += (row * row_stride);
    vals += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = residual[i * iteration_stride + id];
        sum += vals_arr[i];
    }
    if (high_index < row_stride) {
        vals_arr[iterations] = residual[high_index];
        sum += vals_arr[iterations];
        iterations++;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

#if !defined(__STOCHASTIC_MODE__) || __CUDA_ARCH__ < 700
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }

    sum = g.shfl(sum, 0);
    float mean = sum / row_stride;
    if (training)
        if (g.thread_rank() == 0) means[row] = mean;
    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] -= mean;
        variance += vals_arr[i] * vals_arr[i];
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= row_stride;
    variance += epsilon;
    if (training)
        if (g.thread_rank() == 0) vars[row] = variance;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        vals_arr[i] = vals_arr[i] * rsqrtf(variance);
        vals_arr[i] =
            vals_arr[i] * gamma[i * iteration_stride + id] + beta[i * iteration_stride + id];
        vals[i * iteration_stride + id] = vals_arr[i];
    }
    if ((high_index) < row_stride) {
        vals_arr[iterations] = vals_arr[iterations] * rsqrtf(variance);
        vals_arr[iterations] = vals_arr[iterations] * gamma[high_index] + beta[high_index];
        vals[high_index] = vals_arr[iterations];
    }
}

__global__ void fused_bias_residual_layer_norm(__half* vals,
                                               const __half* residual,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               bool preLayerNorm,
                                               bool training,
                                               __half* vars,
                                               __half* means,
                                               int row_stride)
{
#if __CUDA_ARCH__ >= 700
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;

    float2 vals_f[NORM_REG];
    __shared__ float shr[MAX_WARP_NUM];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);

    residual_cast += (row * row_stride);
    vals_cast += (row * row_stride);

    float sum = 0.f;
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        vals_f[i] = __half22float2(residual_cast[i * iteration_stride + id]);
        sum += vals_f[i].x;
        sum += vals_f[i].y;
    }
    if ((high_index) < row_stride) {
        vals_f[iterations] = __half22float2(residual_cast[high_index]);
        sum += vals_f[iterations].x;
        sum += vals_f[iterations].y;
        iterations++;
    }

    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) shr[gid] = sum;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) sum = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { sum += g.shfl_down(sum, i); }
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride * 2);

    float variance = 0.f;
    for (int i = 0; i < iterations; i++) {
        vals_f[i].x -= mean;
        vals_f[i].y -= mean;
        variance += vals_f[i].x * vals_f[i].x;
        variance += vals_f[i].y * vals_f[i].y;
    }

    for (int i = 1; i < 32; i *= 2) { variance += g.shfl_down(variance, i); }

    if (g.thread_rank() == 0) shr[gid] = variance;

    b.sync();

    if (g.thread_rank() < (iteration_stride >> 5)) variance = shr[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    b.sync();
#endif

    for (int i = 1; i < (iteration_stride >> 5); i *= 2) { variance += g.shfl_down(variance, i); }
    variance = g.shfl(variance, 0);
    variance /= (row_stride * 2);
    variance += epsilon;

    __half2 variance_h = __float2half2_rn(variance);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);

    if (training && g.thread_rank() == 0) {
        vars[row] = __float2half(variance);
        means[row] = __float2half(mean);
    }
    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) {
        __half2 vals_arr = __float22half2_rn(vals_f[i]);
        vals_arr = vals_arr * h2rsqrt(variance_h);
        vals_arr =
            vals_arr * gamma_cast[i * iteration_stride + id] + beta_cast[i * iteration_stride + id];
        vals_cast[i * iteration_stride + id] = vals_arr;
    }
    if ((high_index) < row_stride) {
        __half2 vals_arr = __float22half2_rn(vals_f[iterations]);
        vals_arr = vals_arr * h2rsqrt(variance_h);
        vals_arr = vals_arr * gamma_cast[high_index] + beta_cast[high_index];
        vals_cast[high_index] = vals_arr;
    }
#endif
}


template <typename T>
void launch_bias_residual_layer_norm(T* vals,
                                     const T* residual,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t* stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means,
                                     int q_index);

template <>
void launch_bias_residual_layer_norm<float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int batch_size,
                                            int hidden_dim,
                                            cudaStream_t* stream,
                                            bool preLayerNorm,
                                            bool training,
                                            float* vars,
                                            float* means,
                                            int q_index)
{
    int threads = THREADS;

    dim3 grid_dim(batch_size);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim(threads);

    #if DEBUG
        std::cout << "queue_index=" << q_index << "\x1b[41;1mlbrf<<<>>>\x1b[0m";
        std::cout << "\x1b[31;1m, vals=" << vals; 
        std::cout << "\x1b[32;1m, residual=" << residual;
        std::cout << "\x1b[33;1m, gamma=" << gamma;
        std::cout << "\x1b[34;1m, betta=" << beta << "\x1b[0m;" << std::endl;
    #endif
    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream[q_index]>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means, hidden_dim);
}

template <>
void launch_bias_residual_layer_norm<__half>(__half* vals,
                                             const __half* residual,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             int batch_size,
                                             int hidden_dim,
                                             cudaStream_t* stream,
                                             bool preLayerNorm,
                                             bool training,
                                             __half* vars,
                                             __half* means,
                                             int q_index)
{
    int threads = 128;

    dim3 grid_dim(batch_size);

    if (hidden_dim > 8192 && hidden_dim <= 16384)
        threads <<= 1;
    else if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 2;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 3;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim(threads);
    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream[q_index]>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means, hidden_dim / 2);
}

template <typename T>
__global__ void LayerNormBackward1(const T* __restrict__ out_grad,
                                   const T* __restrict__ vals_hat,
                                   const T* __restrict__ gamma,
                                   const T* __restrict__ betta,
                                   T* __restrict__ gamma_grad,
                                   T* __restrict__ betta_grad,
                                   int rows,
                                   int width,
                                   bool invertible)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    float betta_reg = (invertible ? (float)betta[idx] : 0.0f);
    float gamma_reg = (float)gamma[idx];

    // Loop across matrix height
    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (invertible ? ((float)vals_hat[offset] - betta_reg) / gamma_reg
                                : (float)vals_hat[offset]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        int pos = blockIdx.x * TILE_DIM + threadIdx.y;
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

/* Normalize Gamma & Betta gradients
 * Compute gradients using the input to
 * the normalize.
 * Combine transpose with gradients computation.
 */

template <typename T>
__global__ void LayerNormBackward1(const T* __restrict__ out_grad,
                                   const T* __restrict__ X_data,
                                   const T* __restrict__ vars,
                                   const T* __restrict__ means,
                                   T* __restrict__ gamma_grad,
                                   T* __restrict__ betta_grad,
                                   int rows,
                                   int width)
{
    __shared__ float betta_buffer[TILE_DIM][TILE_DIM + 1];
    __shared__ float gamma_buffer[TILE_DIM][TILE_DIM + 1];

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = threadIdx.y * width + idx;
    int y_stride = width * TILE_DIM;

    int pos = blockIdx.x * TILE_DIM + threadIdx.y;
    // Loop across matrix height

    float betta_tmp = 0;
    float gamma_tmp = 0;
    for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        float grad = (float)out_grad[offset];
        float val = (float)X_data[offset];
        val = (val - (float)means[r]) * rsqrtf((float)vars[r]);
        betta_tmp += grad;
        gamma_tmp += (val * grad);

        offset += y_stride;
    }

    betta_buffer[threadIdx.x][threadIdx.y] = betta_tmp;
    gamma_buffer[threadIdx.x][threadIdx.y] = gamma_tmp;

    __syncthreads();

    // Sum the shared buffer.
    float s1 = betta_buffer[threadIdx.y][threadIdx.x];
    float s2 = gamma_buffer[threadIdx.y][threadIdx.x];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < TILE_DIM; i <<= 1) {
        s1 += g.shfl_down(s1, i);
        s2 += g.shfl_down(s2, i);
    }

    if (threadIdx.x == 0) {
        betta_grad[pos] = s1;
        gamma_grad[pos] = s2;
    }
}

__global__ void LayerNormBackward2(const float* out_grad,
                                   const float* X_vals,
                                   const float* gamma,
                                   const float* vars,
                                   const float* means,
                                   float* inp_grad,
                                   int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[MAX_WARP_NUM];

    out_grad += (row * row_stride);
    X_vals += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad[high_index];
        vals_arr[iterations] *= gamma_reg;
        iterations++;
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[NORM_REG];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (X_vals[i * iteration_stride + id] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) inp_grad[i * iteration_stride + id] = (vals_arr[i] - sum);
    if ((high_index) < row_stride) inp_grad[high_index] = (vals_arr[iterations] - sum);
}

/* Backward Normalize (Input-Gradient)
 * Using the means and variances from the input
 * This type of backward is invertible!
 * We do the backward using the X_hat (X - u) / sqrt(variance) or the output of Normalization.
 */

__global__ void LayerNormBackward2(const float* out_grad,
                                   const float* vals_hat,
                                   const float* gamma,
                                   const float* betta,
                                   const float* vars,
                                   float* inp_grad,
                                   bool invertible,
                                   int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[MAX_WARP_NUM];

    out_grad += (row * row_stride);
    vals_hat += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible ? (vals_hat[i * iteration_stride + id] - betta[i * iteration_stride + id]) /
                              gamma_reg
                        : vals_hat[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat[high_index] - betta[high_index]) / gamma_reg
                        : vals_hat[high_index]);
        iterations++;
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += vals_hat_arr[i] * vals_arr[i] *
               sqrtf(var_reg);           // dval_hat = gamma * (x - u) * out_grad
        vals_arr[i] *= rsqrtf(var_reg);  // dvar_inv = gamma * out_grad / sqrt(var)
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++) inp_grad[i * iteration_stride + id] = (vals_arr[i] - sum);
    if ((high_index) < row_stride) inp_grad[high_index] = (vals_arr[iterations] - sum);

    for (int i = 0; i < iterations; i++)
        printf("%f ", inp_grad[i]);

    printf("\n");
}

__global__ void LayerNormBackward2_fused_add(const float* out_grad1,
                                             const float* out_grad2,
                                             const float* X_vals,
                                             const float* gamma,
                                             const float* vars,
                                             const float* means,
                                             float* inp_grad,
                                             int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[MAX_WARP_NUM];

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];

    out_grad1 += (row * row_stride);
    out_grad2 += (row * row_stride);
    X_vals += (row * row_stride);
    inp_grad += (row * row_stride);
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad1[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] = X_vals[i * iteration_stride + id];
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad1[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] = X_vals[high_index];
        iterations++;
    }

    float var_reg = vars[row];
    float mean_reg = means[row];

    float sum = 0;
    float xu[NORM_REG];
    for (int i = 0; i < iterations; i++) {
        xu[i] = (vals_hat_arr[i] - mean_reg);
        sum += vals_arr[i] * xu[i];
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) {
        vals_arr[i] += (-sum * xu[i] * rsqrtf(var_reg) / (var_reg));
    }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        inp_grad[i * iteration_stride + id] =
            (vals_arr[i] - sum) + out_grad2[i * iteration_stride + id];
    if ((high_index) < row_stride)
        inp_grad[high_index] = (vals_arr[iterations] - sum) + out_grad2[high_index];
}

__global__ void LayerNormBackward2_fused_add(const float* out_grad1,
                                             const float* out_grad2,
                                             const float* vals_hat,
                                             const float* gamma,
                                             const float* betta,
                                             const float* vars,
                                             float* inp_grad,
                                             bool invertible,
                                             int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int wid = id / WARP_SIZE;
    int warp_num = (THREADS < row_stride ? THREADS : row_stride) / WARP_SIZE;
    __shared__ float partialSum[MAX_WARP_NUM];

    out_grad1 += (row * row_stride);
    out_grad2 += (row * row_stride);
    vals_hat += (row * row_stride);
    inp_grad += (row * row_stride);

    float vals_arr[NORM_REG];
    float vals_hat_arr[NORM_REG];
    int high_index = iterations * iteration_stride + id;
#pragma unroll
    for (int i = 0; i < iterations; i++) {
        float gamma_reg = gamma[i * iteration_stride + id];
        vals_arr[i] = out_grad1[i * iteration_stride + id];
        vals_arr[i] *= gamma_reg;
        vals_hat_arr[i] =
            (invertible ? (vals_hat[i * iteration_stride + id] - betta[i * iteration_stride + id]) /
                              gamma_reg
                        : vals_hat[i * iteration_stride + id]);
    }
    if ((high_index) < row_stride) {
        float gamma_reg = gamma[high_index];
        vals_arr[iterations] = out_grad1[high_index];
        vals_arr[iterations] *= gamma_reg;
        vals_hat_arr[iterations] =
            (invertible ? (vals_hat[high_index] - betta[high_index]) / gamma_reg
                        : vals_hat[high_index]);
        iterations++;
    }

    float var_reg = vars[row];

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += vals_hat_arr[i] * vals_arr[i] * sqrtf(var_reg);
        vals_arr[i] *= rsqrtf(var_reg);
    }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);

    sum = g.shfl(sum, 0);
    sum /= row_stride;

    for (int i = 0; i < iterations; i++) { vals_arr[i] += ((-sum * vals_hat_arr[i]) / var_reg); }

    sum = 0;
    for (int i = 0; i < iterations; i++) { sum += vals_arr[i]; }

    for (int i = 1; i < WARP_SIZE; i *= 2) { sum += g.shfl_down(sum, i); }

    if (g.thread_rank() == 0) partialSum[wid] = sum;

    __syncthreads();

    if (g.thread_rank() < warp_num) sum = partialSum[g.thread_rank()];

#ifndef __STOCHASTIC_MODE__
    __syncthreads();
#endif

    for (int i = 1; i < warp_num; i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= row_stride;

    iterations = row_stride / iteration_stride;
    for (int i = 0; i < iterations; i++)
        inp_grad[i * iteration_stride + id] =
            (vals_arr[i] - sum) + out_grad2[i * iteration_stride + id];
    if ((high_index) < row_stride)
        inp_grad[high_index] = (vals_arr[iterations] - sum) + out_grad2[high_index];
}

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                                      const T* X_data,
                                      const T* vars,
                                      const T* means,
                                      const T* gamma,
                                      T* gamma_grad,
                                      T* betta_grad,
                                      T* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2]);

template <>
void launch_layerNorm_backward<float>(const float* out_grad,
                                      const float* X_data,
                                      const float* vars,
                                      const float* means,
                                      const float* gamma,
                                      float* gamma_grad,
                                      float* betta_grad,
                                      float* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2])
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);
    LayerNormBackward2<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad, X_data, gamma, vars, means, inp_grad, hidden_dim);
}

template <typename T>
void launch_layerNorm_backward(const T* out_grad,
                                      const T* vals_hat,
                                      const T* vars,
                                      const T* gamma,
                                      T* gamma_grad,
                                      T* betta_grad,
                                      T* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2],
                                      bool invertible,
                                      const T* betta);

template <>
void launch_layerNorm_backward<float>(const float* out_grad,
                                      const float* vals_hat,
                                      const float* vars,
                                      const float* gamma,
                                      float* gamma_grad,
                                      float* betta_grad,
                                      float* inp_grad,
                                      int batch,
                                      int hidden_dim,
                                      cudaStream_t stream[2],
                                      bool invertible,
                                      const float* betta)
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);

    LayerNormBackward2<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad, vals_hat, gamma, betta, vars, inp_grad, invertible, hidden_dim);
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
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2]);

template <>
void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                const float* out_grad2,
                                                const float* X_data,
                                                const float* vars,
                                                const float* means,
                                                const float* gamma,
                                                float* gamma_grad,
                                                float* betta_grad,
                                                float* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2])
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);

    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, X_data, vars, means, gamma_grad, betta_grad, batch, hidden_dim);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);
    LayerNormBackward2_fused_add<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad1, out_grad2, X_data, gamma, vars, means, inp_grad, hidden_dim);
}

template <typename T>
void launch_layerNorm_backward_fused_add(const T* out_grad1,
                                                const T* out_grad2,
                                                const T* vals_hat,
                                                const T* vars,
                                                const T* gamma,
                                                T* gamma_grad,
                                                T* betta_grad,
                                                T* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2],
                                                bool invertible,
                                                const T* betta);

template <>
void launch_layerNorm_backward_fused_add<float>(const float* out_grad1,
                                                const float* out_grad2,
                                                const float* vals_hat,
                                                const float* vars,
                                                const float* gamma,
                                                float* gamma_grad,
                                                float* betta_grad,
                                                float* inp_grad,
                                                int batch,
                                                int hidden_dim,
                                                cudaStream_t stream[2],
                                                bool invertible,
                                                const float* betta)
{
    int threads = THREADS;

    dim3 grid_dim(hidden_dim / TILE_DIM);
    dim3 block_dim(TILE_DIM, TILE_DIM);
    LayerNormBackward1<float><<<grid_dim, block_dim, 0, stream[0]>>>(
        out_grad1, vals_hat, gamma, betta, gamma_grad, betta_grad, batch, hidden_dim, invertible);

    dim3 grid_dim2(batch);

    if (hidden_dim > 16384 && hidden_dim <= 32768)
        threads <<= 1;
    else if (hidden_dim > 32768 && hidden_dim <= 65536)
        threads <<= 2;
    else if (hidden_dim > 65536)
        throw std::runtime_error("Unsupport hidden_dim.");

    dim3 block_dim2(threads);
    LayerNormBackward2_fused_add<<<grid_dim2, block_dim2, 0, stream[1]>>>(
        out_grad1, out_grad2, vals_hat, gamma, betta, vars, inp_grad, invertible, hidden_dim);
}

template <typename T>
class Normalize {
public:
    struct Config {
        uint32_t batchSize;
        uint32_t seqLength;
        uint32_t hiddenDim;
        float epsilon;
        bool training;
        bool useMean;
        Config(uint32_t batch, uint32_t seq, uint32_t h, bool training, bool useMean = true)
            : batchSize(batch),
              seqLength(seq),
              hiddenDim(h),
              epsilon(1e-12),
              training(training),
              useMean(useMean)
        {
        }
    };

    Normalize(Config config)
        : config_(config), vars(nullptr), means(nullptr), vals_hat(nullptr)
    {
    }

    ~Normalize() {}
    void ForwardCheckpoint(int bsz,  // batch * seq
                           Buffer<T>* vals,
                           Buffer<T>* residual,
                           Buffer<T>* gamma,
                           Buffer<T>* betta,
                           ScheduleEngine* SE,
                           bool sync = false,
                           bool preLayerNorm = false)
    {
        #if EVENT_PROFILE
        Stopwatch sw;
        sw.restart();
#endif 
        vals->copyH2D(SE->compute);
        residual->copyH2D(SE->compute);
        gamma->copyH2D(SE->compute);
        betta->copyH2D(SE->compute);
#if EVENT_PROFILE
        sw.stop();
        printf("H2D Time:%lf\n",sw.GetTimeInSeconds());
        sw.restart();
#endif
        launch_bias_residual_layer_norm(vals->get_device_data(),
                                        residual->get_device_data(),
                                        gamma->get_device_data(),
                                        betta->get_device_data(),
                                        config_.epsilon,
                                        bsz,
                                        config_.hiddenDim,
                                        SE->compute,
                                        preLayerNorm,
                                        config_.training,
                                        vars->get_device_data(),
                                        means->get_device_data(),
                                        0);

#if EVENT_PROFILE
        sw.stop();
        printf("Kernel Time:%lf\n",sw.GetTimeInSeconds());
        sw.restart();
#endif
        vals->copyD2H(SE->compute);
        residual->copyD2H(SE->compute);
        // gamma->copyD2H(SE->compute);
        // betta->copyD2H(SE->compute);

        printf("Pushpinder -> vals host data: %lf\n", vals->get_host_data());
        printf("Pushpinder -> vals device data: %lf\n", vals->get_device_data());
        printf("Pushpinder -> vals no of elements: %lf\n", vals->get_num_elements());
        printf("Pushpinder -> vals size: %lf\n\n", vals->get_size());

        printf("Pushpinder -> residual host data: %lf\n", residual->get_host_data());
        printf("Pushpinder -> residual device data: %lf\n", residual->get_device_data());
        printf("Pushpinder -> residual no of elements: %lf\n", residual->get_num_elements());
        printf("Pushpinder -> residual size: %lf\n\n", residual->get_size());
                
        if ( sync )
            CHECK(cudaThreadSynchronize());
#if EVENT_PROFILE
        sw.stop();
        printf("D2H Time:%lf\n",sw.GetTimeInSeconds());
        sw.restart();
#endif


    }

    void ForwardCheckpointPartition(int bsz,  // batch * seq
        int nq, // number of queues
        Buffer<T>* vals,
        Buffer<T>* residual,
        Buffer<T>* gamma,
        Buffer<T>* betta,
        ScheduleEngine* SE,
        bool sync = false,
        bool preLayerNorm = false)
    {
        uint32_t batch_size = config_.batchSize; 
        uint32_t sequence_length = config_.seqLength; 
        uint32_t hidden_size = config_.hiddenDim;
        std::cout << "\x1b[32;1mForwardCheckpointPartition\x1b[0m\n";
        int offset = 0;
        int partition_size = (batch_size / nq);
        std::cout << "\x1b[31;1mpartition_size=" << partition_size << "\x1b[0m" << std::endl;
        gamma->copyH2D(SE->compute);
        betta->copyH2D(SE->compute);    
        std::cout << "creating queues" << std::endl;

        Stopwatch sw;
        sw.start();
        // std::cout << "start profiling" << std::endl;
        for (int i = 0; i < nq; i++)
        {
            offset = i * partition_size * sequence_length * hidden_size; 
            
            vals->copyH2D(SE->compute, offset, nq, i);
            residual->copyH2D(SE->compute, offset, nq, i);
            #if DEBUG
                std::cout << "queue_index=" << i << ", offset=" << offset; 
                std::cout << "\x1b[31;1m, vals=" << vals->get_device_data(offset); 
                std::cout << "\x1b[32;1m, residual=" << residual->get_device_data(offset);
                std::cout << "\x1b[33;1m, gamma=" << gamma->get_device_data();
                std::cout << "\x1b[34;1m, betta=" << betta->get_device_data() << "\x1b[0m;" << std::endl;
            #endif
            cublasSetStream(SE->handle, SE->compute[i]);
            launch_bias_residual_layer_norm(vals->get_device_data(offset),
                                residual->get_device_data(offset),
                                gamma->get_device_data(),
                                betta->get_device_data(),
                                config_.epsilon,
                                bsz,
                                config_.hiddenDim,
                                SE->compute,
                                preLayerNorm,
                                config_.training,
                                vars->get_device_data(),
                                means->get_device_data(),
                                i);
            vals->copyD2H(SE->compute, offset, nq, i);
            // residual->copyD2H(SE->compute, offset, nq, i);
        }
        // sw.stop();
        // std::cout << "end profiling, time=" << sw.GetTimeInSeconds() << std::endl;
        if ( sync )
            CHECK(cudaThreadSynchronize());
    }

    void Backward(int bsz,
                  Buffer<T>* out_grad,
                  Buffer<T>* gamma,
                  Buffer<T>* gamma_grad,
                  Buffer<T>* betta_grad,
                  ScheduleEngine* SE,
                  Buffer<T>* inp_grad_out,
                  Buffer<T>* norm_in = nullptr,
                  bool sync = true)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        out_grad->copyH2D(SE->compute);
        gamma->copyH2D(SE->compute); 
        gamma_grad->copyH2D(SE->compute);
        betta_grad->copyH2D(SE->compute);
        inp_grad_out->copyH2D(SE->compute);
        norm_in->copyH2D(SE->compute);
        
        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif

        launch_layerNorm_backward(out_grad->get_device_data(),
                                  norm_in->get_device_data(),
                                  vars->get_device_data(),
                                  means->get_device_data(),
                                  gamma->get_device_data(),
                                  gamma_grad->get_device_data(),
                                  betta_grad->get_device_data(),
                                  inp_grad_out->get_device_data(),
                                  bsz,
                                  config_.hiddenDim,
                                  SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif

//        out_grad->copyD2H(SE->compute);
        gamma_grad->copyD2H(SE->compute);
        betta_grad->copyD2H(SE->compute);
        inp_grad_out->copyD2H(SE->compute);

        if ( sync == true )
           CHECK(cudaDeviceSynchronize());

        // #if EVENT_PROFILE
        //     float *host_inp_grad_out = inp_grad_out->get_device_data();
           
        //     for(int i = 0; i < inp_grad_out->get_num_elements(); i++) {
        //         printf("%f ", host_inp_grad_out[i]);
        //         // if(host_inp_grad_out[i] == 1.2345)
        //         //     c++;
        //     }
            
        //     printf("\n");
        // #endif

        // #if EVENT_PROFILE
        //     float *host_inp_grad_out = inp_grad_out->get_host_data();
        //     float *host_out_grad = inp_grad_out->get_host_data();
           
        //     for(int i = 0; i < inp_grad_out->get_size(); i++) {
        //         printf("%d", host_inp_grad_out[i]);
        //     }
            
        //     // printf("/n");

        //     // for(int i = 0; i < out_grad->get_size(); i++) {
        //     //     printf("%d", out_grad[i]);
        //     // }                         
        // #endif

        // printf("Pushpinder -> inp_grad_out host data: %lu\n", inp_grad_out->get_host_data());
        // printf("Pushpinder -> inp_grad_out device data: %lu\n", inp_grad_out->get_device_data());
        // printf("Pushpinder -> inp_grad_out no of elements: %lld\n", inp_grad_out->get_num_elements());
        // printf("Pushpinder -> inp_grad_out size: %lld\n\n", inp_grad_out->get_size());

        // printf("Pushpinder -> out_grad host data: %lu\n", out_grad->get_host_data());
        // printf("Pushpinder -> out_grad device data: %lu\n", out_grad->get_device_data());
        // printf("Pushpinder -> out_grad no of elements: %lld\n", out_grad->get_num_elements());
        // printf("Pushpinder -> out_grad size: %lld \n\n\n\n", out_grad->get_size());

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time: %lf\n\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif                                               
    }

    void Backward(int bsz,
                  Buffer<T>* out_grad,
                  Buffer<T>* gamma,
                  Buffer<T>* betta,
                  Buffer<T>* gamma_grad,
                  Buffer<T>* betta_grad,
                  ScheduleEngine* SE,
                  Buffer<T>* inp_grad_out,
                  Buffer<T>* norm_out,
                  bool sync = true)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        out_grad->copyH2D(SE->compute);
        gamma->copyH2D(SE->compute); 
        betta->copyH2D(SE->compute);
        gamma_grad->copyH2D(SE->compute);
        betta_grad->copyH2D(SE->compute);
        inp_grad_out->copyH2D(SE->compute); 
        norm_out->copyH2D(SE->compute); 

        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif

        launch_layerNorm_backward(out_grad->get_device_data(),
                                  norm_out->get_device_data(),
                                  vars->get_device_data(),
                                  gamma->get_device_data(),
                                  gamma_grad->get_device_data(),
                                  betta_grad->get_device_data(),
                                  inp_grad_out->get_device_data(),
                                  bsz,
                                  config_.hiddenDim,
                                  SE->compute,
                                  !config_.useMean,
                                  betta->get_device_data());

        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif

        out_grad->copyD2H(SE->compute);
        inp_grad_out->copyD2H(SE->compute);
        norm_out->copyD2H(SE->compute);

        if ( sync == true )
           CHECK(cudaDeviceSynchronize());

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time: %lf\n\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif           
    }    

    void BackwardFusedAdd(int bsz,
                          Buffer<T>* out_grad1,
                          Buffer<T>* out_grad2,
                          Buffer<T>* gamma,
                          Buffer<T>* gamma_grad,
                          Buffer<T>* betta_grad,
                          ScheduleEngine* SE,
                          Buffer<T>* inp_grad_out,
                          Buffer<T>* norm_in = nullptr)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        out_grad1->copyH2D(SE->compute);
        out_grad2->copyH2D(SE->compute);
        gamma->copyH2D(SE->compute);
        gamma_grad->copyH2D(SE->compute);
        betta_grad->copyH2D(SE->compute);
        inp_grad_out->copyH2D(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif      

        launch_layerNorm_backward_fused_add(out_grad1->get_device_data(),
                                            out_grad2->get_device_data(),
                                            norm_in->get_device_data(),
                                            vars->get_device_data(),
                                            means->get_device_data(),
                                            gamma->get_device_data(),
                                            gamma_grad->get_device_data(),
                                            betta_grad->get_device_data(),
                                            inp_grad_out->get_device_data(),
                                            bsz,
                                            config_.hiddenDim,
                                            SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif      

        gamma_grad->copyD2H(SE->compute);
        betta_grad->copyD2H(SE->compute);
        inp_grad_out->copyD2H(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time: %lf\n\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif              
    }

    void BackwardFusedAdd(int bsz,
                          Buffer<T>* out_grad1,
                          Buffer<T>* out_grad2,
                          Buffer<T>* gamma,
                          Buffer<T>* betta,
                          Buffer<T>* gamma_grad,
                          Buffer<T>* betta_grad,
                          ScheduleEngine* SE,
                          Buffer<T>* inp_grad_out,
                          Buffer<T>* norm_out)
    {
        #if EVENT_PROFILE
            Stopwatch sw;
            sw.restart();
        #endif

        out_grad1->copyH2D(SE->compute);
        out_grad2->copyH2D(SE->compute);
        gamma->copyH2D(SE->compute);
        betta->copyH2D(SE->compute);
        gamma_grad->copyH2D(SE->compute);
        betta_grad->copyH2D(SE->compute);
        inp_grad_out->copyH2D(SE->compute);

        #if EVENT_PROFILE
            sw.stop();
            printf("H2D Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif     

        launch_layerNorm_backward_fused_add(out_grad1->get_device_data(),
                                            out_grad2->get_device_data(),
                                            norm_out->get_device_data(),
                                            vars->get_device_data(),
                                            gamma->get_device_data(),
                                            gamma_grad->get_device_data(),
                                            betta_grad->get_device_data(),
                                            inp_grad_out->get_device_data(),
                                            bsz,
                                            config_.hiddenDim,
                                            SE->compute,
                                            !config_.useMean,
                                            betta->get_device_data());

        #if EVENT_PROFILE
            sw.stop();
            printf("Kernel Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif     

        gamma_grad->copyD2H(SE->compute);
        betta_grad->copyD2H(SE->compute);
        inp_grad_out->copyD2H(SE->compute);  

        #if EVENT_PROFILE
            sw.stop();
            printf("D2H Time: %lf\n", sw.GetTimeInSeconds());
            sw.restart();
        #endif                                                       
    }

    void BackwardFineGrained(int bsz,
                  int nq, // number of queues
                  Buffer<T>* out_grad,
                  Buffer<T>* gamma,
                  Buffer<T>* gamma_grad,
                  Buffer<T>* betta_grad,
                  ScheduleEngine* SE,
                  Buffer<T>* inp_grad_out,
                  Buffer<T>* norm_in = nullptr,
                  bool sync = true)
    {
        uint32_t batch_size = config_.batchSize; 
        uint32_t sequence_length = config_.seqLength; 
        uint32_t hidden_size = config_.hiddenDim;
        int offset = 0;
        int partition_size = (batch_size / nq);
        
        gamma->copyH2D(SE->compute); 
        gamma_grad->copyH2D(SE->compute);
        betta_grad->copyH2D(SE->compute);
        inp_grad_out->copyH2D(SE->compute);

        for (int i = 0; i < nq; i++) {
            offset = i * partition_size * sequence_length * hidden_size;

            out_grad->copyH2D(SE->compute, offset, nq, i);
            norm_in->copyH2D(SE->compute, offset, nq, i);
        
            #if DEBUG
                std::cout << "queue_index=" << i << ", offset=" << offset; 
                std::cout << "\x1b[31;1m, vals=" << vals->get_device_data(offset); 
                std::cout << "\x1b[32;1m, residual=" << residual->get_device_data(offset);
                std::cout << "\x1b[33;1m, gamma=" << gamma->get_device_data();
                std::cout << "\x1b[34;1m, betta=" << betta->get_device_data() << "\x1b[0m;" << std::endl;
            #endif
            
            cublasSetStream(SE->handle, SE->compute[i]);
            
            launch_layerNorm_backward(out_grad->get_device_data(offset),
                                  norm_in->get_device_data(offset),
                                  vars->get_device_data(),
                                  means->get_device_data(),
                                  gamma->get_device_data(),
                                  gamma_grad->get_device_data(),
                                  betta_grad->get_device_data(),
                                  inp_grad_out->get_device_data(),
                                  bsz,
                                  config_.hiddenDim,
                                  SE->compute);
    
            // out_grad->copyD2H(SE->compute, offset, nq, i);
        }
        
        gamma_grad->copyD2H(SE->compute);
        betta_grad->copyD2H(SE->compute);
        inp_grad_out->copyD2H(SE->compute);

        if ( sync == true )
           CHECK(cudaDeviceSynchronize());              
    }

    void BackwardFineGrained(int bsz,
                  int nq, // number of queues
                  Buffer<T>* out_grad,
                  Buffer<T>* gamma,
                  Buffer<T>* betta,
                  Buffer<T>* gamma_grad,
                  Buffer<T>* betta_grad,
                  ScheduleEngine* SE,
                  Buffer<T>* inp_grad_out,
                  Buffer<T>* norm_out,
                  bool sync = true)
    {
        uint32_t batch_size = config_.batchSize; 
        uint32_t sequence_length = config_.seqLength; 
        uint32_t hidden_size = config_.hiddenDim;
        int offset = 0;
        int partition_size = (batch_size / nq);
        
        gamma->copyH2D(SE->compute);
        betta->copyH2D(SE->compute);
        gamma_grad->copyH2D(SE->compute);
        betta_grad->copyH2D(SE->compute);
        inp_grad_out->copyH2D(SE->compute);

        Stopwatch sw;
        sw.start();

        for (int i = 0; i < nq; i++) {
            offset = i * partition_size * sequence_length * hidden_size; 
            
            out_grad->copyH2D(SE->compute, offset, nq, i);
            norm_out->copyH2D(SE->compute, offset, nq, i);

            #if DEBUG
                // std::cout << "queue_index=" << i << ", offset=" << offset; 
                // std::cout << "\x1b[31;1m, vals=" << vals->get_device_data(offset); 
                // std::cout << "\x1b[32;1m, residual=" << residual->get_device_data(offset);
                // std::cout << "\x1b[33;1m, gamma=" << gamma->get_device_data();
                // std::cout << "\x1b[34;1m, betta=" << betta->get_device_data() << "\x1b[0m;" << std::endl;
            #endif
            
            cublasSetStream(SE->handle, SE->compute[i]);

            launch_layerNorm_backward(out_grad->get_device_data(offset),
                                  norm_out->get_device_data(offset),
                                  vars->get_device_data(),
                                  gamma->get_device_data(),
                                  gamma_grad->get_device_data(),
                                  betta_grad->get_device_data(),
                                  inp_grad_out->get_device_data(),
                                  bsz,
                                  config_.hiddenDim,
                                  SE->compute,
                                  !config_.useMean,
                                  betta->get_device_data());          
        }

        betta_grad->copyD2H(SE->compute);
        inp_grad_out->copyD2H(SE->compute);

        float* h = betta_grad->get_host_data();
        
        printf("Pushpinder -> host data: %lf\n", betta_grad->get_host_data());
        printf("Pushpinder -> device data: %lf\n", betta_grad->get_device_data());
        printf("Pushpinder -> no of elements: %lf\n", betta_grad->get_num_elements());
        printf("Pushpinder -> size: %lf\n", betta_grad->get_size());

        if ( sync == true )
           CHECK(cudaDeviceSynchronize());  
    }        

    inline bool UseMean() const { return config_.useMean; }

    inline void SetVar(T* variance)
    {
        if (!variance) { throw std::runtime_error("Normalize variance is null."); }
        vars = variance;
    }

    inline void SetMean(T* mean)
    {
        if (!mean) { throw std::runtime_error("Normalize mean is null."); }
        means = mean;
    }
  
    void SetMeansAndVariance(Buffer<float>*mean, Buffer<float>*variance)
    {
      means = mean;
      vars = variance;
    }

private:
    Config config_;
    Buffer<T>* vars;
    Buffer<T>* means;
    Buffer<T>* vals_hat;

};