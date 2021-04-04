// %%cuda --name ../tests/normalize_layer.cu
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
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <stdexcept>
// #include "json.hpp"

namespace cg = cooperative_groups;
// using json = nlohmann::json;

#define MAX_THREADS 1024
#define THREADS 256
#define DEBUG true // added a flag to turn off prinitng apart from testing

#define MAX_THREAD_STRIDE 32
#define TILE_DIM 32
#define WARP_SIZE 32
// Maximum sequence-length support based on the number of threads (2048) allowed in each block and
// this MAX is 8K For higher sequence length we need to use higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K
#define MAX_WARP_NUM 32
#define MAX_REGISTERS 256
#define NORM_REG (MAX_REGISTERS / 4)


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

    if (DEBUG) 
        std::cout << "queue_index=" << q_index << "\x1b[41;1mlbrf<<<>>>\x1b[0m";
        std::cout << "\x1b[31;1m, vals=" << vals; 
        std::cout << "\x1b[32;1m, residual=" << residual;
        std::cout << "\x1b[33;1m, gamma=" << gamma;
        std::cout << "\x1b[34;1m, betta=" << beta << "\x1b[0m;" << std::endl;
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

    // void from(std::string fname) {
    //     json j;
    //     std::ifstream fin(fname);
    //     fin >> j;
    //     std::vector <T> vec = j;
        
    //     if (vec.size() != num_elements) {
    //         std::cout << "the file has a tensor of different size";
    //         exit(EXIT_FAILURE);
    //     }
        
    //     for (int i = 0; i < num_elements; i++)
    //         _host_data[i] = vec[i];
    // } 

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

    /* void to(std::string fname) {
        std::vector <T> dump_t;
        for (int i = 0; i < num_elements; i++)
            dump_t.push_back(_host_data[i]);
        json j = json::parse(dump_t);
        std::ofstream fout(fname);
        std::cout << j;
        // fout << j.dump();
    } */

    void print_host_data() {
        for (int i = 0; i < num_elements; i++)
            std::cout << _host_data[i] << "\n";
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
// template <> Buffer <float>;

/* template <> void Buffer<float>::from(std::string fname) {
        json j; 
        std::ifstream fin(fname); 
        fin >> j;
        std::vector <float> vec = j;
        
        if (vec.size() != num_elements) {
            std::cout << "the file has a tensor of different size";
            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < num_elements; i++)
            _host_data[i] = vec[i];
} */

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
         
        vals->copyH2D(SE->compute);
        residual->copyH2D(SE->compute);
        gamma->copyH2D(SE->compute);
        betta->copyH2D(SE->compute);

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


        vals->copyD2H(SE->compute);
        residual->copyD2H(SE->compute);
        // gamma->copyD2H(SE->compute);
        // betta->copyD2H(SE->compute);


    }

    void ForwardCheckpointPartition(int bsz,  // batch * seq
        int nq, // number of queues
        Buffer<T>* vals,
        Buffer<T>* residual,
        Buffer<T>* gamma,
        Buffer<T>* betta,
        ScheduleEngine* SE,
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

        for (int i = 0; i<nq; i++)
        {
            offset = i * partition_size * sequence_length * hidden_size; 
    
            //vals->copyH2D(SE->compute, offset, nq, i);
            residual->copyH2D(SE->compute, offset, nq, i);
            if (DEBUG)
                std::cout << "queue_index=" << i << ", offset=" << offset; 
                std::cout << "\x1b[31;1m, vals=" << vals->get_device_data(offset); 
                std::cout << "\x1b[32;1m, residual=" << residual->get_device_data(offset);
                std::cout << "\x1b[33;1m, gamma=" << gamma->get_device_data();
                std::cout << "\x1b[34;1m, betta=" << betta->get_device_data() << "\x1b[0m;" << std::endl;

            cublasSetStream(SE->handle, SE->compute[i]);
            launch_bias_residual_layer_norm(vals->get_device_data(offset),
                                residual->get_device_data(offset),
                                gamma->get_device_data(),
                                betta->get_device_data(),
                                config_.epsilon,
                                bsz/partition_size,
                                config_.hiddenDim,
                                SE->compute,
                                preLayerNorm,
                                config_.training,
                                vars->get_device_data(),
                                means->get_device_data(),
                                i);
            
            vals->copyD2H(SE->compute, offset, nq, i);
            //residual->copyD2H(SE->compute, offset, nq, i);
        }
	CHECK(cudaThreadSynchronize());

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
    // PLEASE NOTE: number of queues must be less than batch_size
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int nq = atoi(argv[4]);
 
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
    printf("Read command line parameters\n");
 
    ScheduleEngine SE(nq);
    // input.print_host_data();
    // input.to("src/test.json");
    
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> input_norm(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> norm_weights(hidden_size, &SE);
    norm_weights.from("../dump/encoder.layer.0.output.LayerNorm.weight.json");
    Buffer<float> norm_bias(hidden_size, &SE);
    norm_bias.from("../dump/encoder.layer.0.output.LayerNorm.bias.json");
    Buffer<float> norm_var(batch_size * sequence_length, &SE);
    Buffer<float> norm_mean(batch_size * sequence_length, &SE);

    float layernorm_eps=0.000001; 
    Normalize<float> normalize_input(Normalize<float>::Config(batch_size , sequence_length, hidden_size, layernorm_eps,true));
    normalize_input.SetMeansAndVariance(&norm_mean,&norm_var);
    //normalize_input.ForwardCheckpoint(batch_size*sequence_length,&input_norm,&input,&norm_weights,&norm_bias,&SE);
    normalize_input.ForwardCheckpointPartition(batch_size*sequence_length, nq, &input_norm, &input, &norm_weights, &norm_bias, &SE);
    input_norm.to("../dump/input_norm.json");
    printf("Executed normalize layer\n"); 
}
