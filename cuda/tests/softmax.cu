// %%cuda --name Softmax.cu

#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
// #include "custom_cuda_layers.h"
// #include "cublas_wrappers.h"

namespace cg = cooperative_groups;

#define MAX_THREADS 1024
#define THREADS 256
#define DEBUG false // added a flag to turn off prinitng apart from testing

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32
#define WARP_SIZE 32
// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32
#define MAX_REGISTERS 256
#define NORM_REG (MAX_REGISTERS / 4)
#define NUM_STREAMS 32

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

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


class Stopwatch {
  private:
    float m_total_time;
    struct timespec m_start_time;
    bool m_is_started;

  public:
    Stopwatch() {
        m_total_time = 0.0;
        m_is_started = false;
    }

    ~Stopwatch() {}

    void Reset() { m_total_time = 0.0; }

    void start() {
        clock_gettime(CLOCK_MONOTONIC, &m_start_time);
        m_is_started = true;
    }

    void restart() {
        m_total_time = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &m_start_time);
        m_is_started = true;
    }

    void stop() {
        if (m_is_started) {
            m_is_started = false;

            struct timespec end_time;
            clock_gettime(CLOCK_MONOTONIC, &end_time);

            m_total_time +=
                (float)(end_time.tv_sec - m_start_time.tv_sec) +
                (float)(end_time.tv_nsec - m_start_time.tv_nsec) / 1e9;
        }
    }

    float GetTimeInSeconds() {
        if (m_is_started) {
            stop();
            start();
        }
        return m_total_time;
    }
};

class ScheduleEngine {
  public:
    ScheduleEngine(int num_queues) {
    assert(num_queues < NUM_STREAMS);
    CHECK_CUBLAS(cublasCreate(&handle));
    this->num_queues = num_queues;
    compute = (cudaStream_t *) malloc(num_queues * sizeof(cudaStream_t));
    for(int i=0;i<num_queues;i++)
    {
        CHECK(cudaStreamCreate(&compute[i]));
    }

   }

   ~ScheduleEngine() { delete compute; }

    int num_queues;
    cublasHandle_t handle;
    cudaStream_t *compute;
    cudaStream_t& getStream(int idx){return compute[idx];}
   
};

template <typename T>
class Buffer {
  public:
    Buffer(int num_elements, ScheduleEngine *se) {
        this->num_elements = num_elements;
        printf("Creating host data\n");
        CHECK(cudaMallocHost((void **)&_host_data, sizeof(T)*num_elements));
        printf("Creating device data\n");
        CHECK(cudaMalloc((void **)&_device_data, sizeof(T)*num_elements));
        printf("Initializing host data\n");
        init_ones();
        printf("Finished creating Buffer\n");
    }
    T *get_host_data() { return _host_data; }
    T *get_host_data(int offset) { return _host_data + offset; }
    T *get_device_data() { return _device_data; }
    T *get_device_data(int offset) { return _device_data + offset; }

    size_t get_size() { return sizeof(T) * num_elements; }
    size_t get_size(int nq) { return sizeof(T) * (num_elements/nq); }
    int get_num_elements() { return num_elements; }

    ~Buffer() {}

    void init_ones() {
        for (int i = 0; i < num_elements; i++)
            _host_data[i] = 1;
    }

    void from(std::string fname) {
        std::ifstream fin(fname);
        std::string word;
        
        int i = 0;
        while ( fin >> word ) {
            if ( word.rfind("[",0) == 0 ) 
                word = word.replace(0,1,"");
            if ( word.compare(word.size()-1, 1, ",") == 0 )
                word = word.replace(word.length()-1, word.length(),"");
            if ( word.compare(word.size()-1, 1, "]") == 0 )
                word = word.replace(word.length()-1, word.length(),"");

            _host_data[i] = stof(word); 
            i++;
            // std::cout << stof(word) << std::endl; 
        }
        fin.close();
    }

    void to(std::string fname) {
        std::ofstream fout(fname);
        fout << "[";
        for (int i = 0; i < num_elements-1; i++) {
            fout << _host_data[i] << ", ";
        }
        fout << _host_data[num_elements-1];
        fout << "]";
    }

    void random(void) {
        std::srand (static_cast <unsigned> (std::time(0)));
        for ( int i = 0; i < num_elements; i++ ) {
            _host_data[i] = static_cast <float> (std::rand())/static_cast <float> (RAND_MAX);
        }
    }

    void binarize(float thresh=0.5) {
        #if DEBUG
        std::cout << "using thresh=" << thresh << std::endl;
        #endif
        for ( int i = 0; i < num_elements; i++ ) {
            #if DEBUG
                std::cout << "bin(" << _host_data[i] << "," << thresh << ") =";
            #endif
            _host_data[i] = (( _host_data[i] > thresh ) ? 1 : 0);
            #if DEBUG
                std::cout << _host_data[i] << std::endl;
            #endif
        }
    }

    void print_host_data(int start=0, int end=-1) {
        #if DEBUG
            std::cout << "\x1b[31;1minside print_host_data\x1b[0m" << std::endl; 
            std::cout << "num_elements=" << num_elements << std::endl;
            std::cout << "start=" << start << ", end=" << end << std::endl;
        #endif 
        if ( start < 0 ) 
            start = num_elements + start;
        if ( end < 0 ) 
            end = num_elements + end;
        if ( start < 0 )
            start = 0;
        if ( end < 0 )
            end = 0;
        // std::cout << "DEBUG=" << DEBUG << std::endl;
        #if DEBUG
            std::cout << "start=" << start << ", end=" << end << std::endl;
        #endif        

        for ( int i = 0; i < num_elements; i++ ) {
            if ( i >= start && i <= end )
                std::cout << _host_data[i] << "\n";
        }
    }
    
    void copyD2H(cudaStream_t *q, int offset=0)
    {
      T *h = get_host_data(offset);
      T *d = get_device_data(offset);
      CHECK(cudaMemcpyAsync(h, d, get_size(),cudaMemcpyDeviceToHost, q[0]));
    }

    void copyH2D(cudaStream_t *q, int offset=0)
    {
      T *h = get_host_data(offset);
      T *d = get_device_data(offset);
      CHECK(cudaMemcpyAsync(d, h, get_size(), cudaMemcpyHostToDevice, q[0]));
    }

    void copyD2H(cudaStream_t *q, int offset, int nq, int q_index)
    {
      T *h = get_host_data(offset);
      T *d = get_device_data(offset);
      CHECK(cudaMemcpyAsync(h, d, get_size(nq), cudaMemcpyDeviceToHost, q[q_index]));
    }

    void copyH2D(cudaStream_t *q, int offset, int nq, int q_index)
    {
      T *h = get_host_data(offset);
      T *d = get_device_data(offset);
      CHECK(cudaMemcpyAsync(d, h, get_size(nq), cudaMemcpyHostToDevice, q[q_index]));
    }

  private:
    T *_host_data;
    T *_device_data;
    int num_elements;
};

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
        vals->copyH2D(SE->compute);
        attn_mask->copyH2D(SE->compute);
        launch_attn_softmax<T>(vals->get_device_data(), 
                               attn_mask->get_device_data(), 
                               bsz, 
                               config_.heads, 
                               config_.seq_length, 
                               SE->getStream(q_index));
        vals->copyD2H(SE->compute);
    }
    /*
    void ForwardCheckpointPartition(int bsz, Buffer <T>* vals, Buffer <T>* attn_mask, ScheduleEngine* SE, int q_index)
    {
        int offset = 
        vals->copyH2D(offset);
        vals->copyH2D(offset);
        launch_attn_softmax<T>(vals, attn_mask, bsz, config_.heads, config_.seq_length, SE->compute[q_index]);
    }
    */
    inline size_t GetProbDepth() const { return config_.prob_depth; }

    inline size_t GetBatchSize() const { return config_.batchSize; }

    inline size_t GetNumHeads() const { return config_.heads; }

    inline size_t GetSeqLength() const { return config_.seq_length; }

    inline void SetSeqLength(size_t seq_len) { config_.seq_length = seq_len; }

private:
    Config config_;
};


int main(int argc, char *argv[])
{
    // PLEASE NOTE: number of queues must be less than batch_size
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int nh = atoi(argv[3]);
    int nq = atoi(argv[4]);
    int bsz = batch_size * sequence_length;

    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "hidden layer size=" << sequence_length << std::endl;
    std::cout << "number of heads=" << nh << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
    printf("Read command line parameters\n");
 
    ScheduleEngine SE(nq);
    Buffer<float> soft_out(bsz * nh * sequence_length *sequence_length, &SE);
    Buffer<float> input_mask(bsz * nh * sequence_length * sequence_length, &SE);
    Softmax<float> _softmax(Softmax<float>::Config(batch_size, nh, sequence_length));
    
    for ( int i = 0; i < nq; i++ )
        _softmax.ForwardCheckpoint(bsz, &soft_out, &input_mask, &SE, i);
    // soft_out.print_host_data(-50, -1);
    CHECK(cudaThreadSynchronize());
    soft_out.to("../dumps/softmax_output.json");
    printf("Executed Softmax\n");
}