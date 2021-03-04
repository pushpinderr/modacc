#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include<stdexcept>
namespace cg = cooperative_groups;


#define MAX_THREADS 1024
#define THREADS 256

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32
#define WARP_SIZE 32
// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32
#define MAX_REGISTERS 256
#define NORM_REG (MAX_REGISTERS / 4)


template <typename T>
void copyH2D(T *src, T *dst, size_t size_in_bytes, cudaStream_t q)
{
    CHECK(cudaMemcpyAsync(&src, &dst, size_in_bytes,
                           cudaMemcpyHostToDevice, q));

}
template <typename T>
void copyD2H(T *src, T *dst, size_t size_in_bytes, cudaStream_t q)
{
    CHECK(cudaMemcpyAsync(&src, &dst, size_in_bytes,
                           cudaMemcpyDeviceToHost, q));

}


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
                                     cudaStream_t stream,
                                     bool preLayerNorm,
                                     bool training,
                                     T* vars,
                                     T* means);

template <>
void launch_bias_residual_layer_norm<float>(float* vals,
                                            const float* residual,
                                            const float* gamma,
                                            const float* beta,
                                            float epsilon,
                                            int batch_size,
                                            int hidden_dim,
                                            cudaStream_t stream,
                                            bool preLayerNorm,
                                            bool training,
                                            float* vars,
                                            float* means)
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

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
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
                                             cudaStream_t stream,
                                             bool preLayerNorm,
                                             bool training,
                                             __half* vars,
                                             __half* means)
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

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        vals, residual, gamma, beta, epsilon, preLayerNorm, training, vars, means, hidden_dim / 2);
}


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
    assert(num_queues<NUM_STREAMS);
    CHECK_CUBLAS(cublasCreate(&handle));
        this->num_queues = num_queues;
   }

   ~ScheduleEngine() {}

    int num_queues;
    cublasHandle_t handle;
    cudaStream_t compute[NUM_STREAMS];
    cudaStream_t* getStream(int idx){return &compute[idx];}
   
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
    int get_num_elements() { return num_elements; }

    ~Buffer() {}

    void init_ones() {
        for (int i = 0; i < num_elements; i++)
            _host_data[i] = 1;
    }

    void print_host_data() {
        for (int i = 0; i < num_elements; i++)
            std::cout << _host_data[i] << "\n";
    }
    
    void copyD2H(cudaStream_t *q, int offset=0)
    {
      T *h = get_host_data(offset);
      T *d = get_device_data(offset);
      CHECK(cudaMemcpyAsync(&h, &d, get_size(),cudaMemcpyDeviceToHost, *q));
    }

    void copyH2D(cudaStream_t *q, int offset=0)
    {
      T *h = get_host_data(offset);
      T *d = get_device_data(offset);
      CHECK(cudaMemcpyAsync(&d, &h, get_size(), cudaMemcpyHostToDevice, *q));
    }


  private:
    T *_host_data;
    T *_device_data;
    int num_elements;
};

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
                           bool preLayerNorm = false)
    {
         
        vals->copyH2D(SE->getStream(0));
        residual->copyH2D(SE->getStream(0));
        gamma->copyH2D(SE->getStream(0));
        betta->copyH2D(SE->getStream(0));

        launch_bias_residual_layer_norm(vals->get_device_data(),
                                        residual->get_device_data(),
                                        gamma->get_device_data(),
                                        betta->get_device_data(),
                                        config_.epsilon,
                                        bsz,
                                        config_.hiddenDim,
                                        *(SE->getStream(0)),
                                        preLayerNorm,
                                        config_.training,
                                        vars->get_device_data(),
                                        means->get_device_data());


        vals->copyD2H(SE->getStream(0));
        residual->copyD2H(SE->getStream(0));
        gamma->copyD2H(SE->getStream(0));
        betta->copyD2H(SE->getStream(0));


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


int main(int argc, char *argv[])
{
  
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int nq = atoi(argv[4]);
    printf("Read command line parameters\n");
    ScheduleEngine SE(nq);
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> input_norm(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> norm_weights(hidden_size, &SE);
    Buffer<float> norm_bias(hidden_size, &SE);
    Buffer<float> norm_var(batch_size * sequence_length, &SE);
    Buffer<float> norm_mean(batch_size * sequence_length, &SE);
    float layernorm_eps=0.000001; 
    Normalize<float> normalize_input(Normalize<float>::Config(batch_size , sequence_length, hidden_size, layernorm_eps,true));
    normalize_input.SetMeansAndVariance(&norm_mean,&norm_var);
    normalize_input.ForwardCheckpoint(batch_size*sequence_length,&input_norm,&input,&norm_weights,&norm_bias,&SE);
    printf("Executed normalize layer\n");
}
