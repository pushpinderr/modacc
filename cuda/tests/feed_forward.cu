// %%cuda --name ../include/FeedForward.cu
#include "json.hpp"

#include <cuda.h>
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
using json = nlohmann::json;

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

int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const float* A,
                   const float* B,
                   float* C,
                   cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         CUDA_R_32F,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         CUDA_R_32F,
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         C,
                                         CUDA_R_32F,
                                         m,
                                         CUDA_R_32F,
                                         algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_gemm_ex(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float* alpha,
                   const float* beta,
                   const __half* A,
                   const __half* B,
                   __half* C,
                   cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmEx(handle,
                                         transa,
                                         transb,
                                         m,
                                         n,
                                         k,
                                         (const void*)alpha,
                                         (const void*)A,
                                         CUDA_R_16F,
                                         (transa == CUBLAS_OP_N) ? m : k,
                                         (const void*)B,
                                         CUDA_R_16F,
                                         (transb == CUBLAS_OP_N) ? k : n,
                                         (const void*)beta,
                                         (void*)C,
                                         CUDA_R_16F,
                                         m,
                                         CUDA_R_32F,
                                         algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const float* A,
                                const float* B,
                                float* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       CUDA_R_32F,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       CUDA_R_32F,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       CUDA_R_32F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, error: %d) \n",
                batch,
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }
    return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float* alpha,
                                const float* beta,
                                const __half* A,
                                const __half* B,
                                __half* C,
                                cublasOperation_t op_A,
                                cublasOperation_t op_B,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo)
{
    cublasStatus_t status = cublasGemmStridedBatchedEx(handle,
                                                       op_A,
                                                       op_B,
                                                       m,
                                                       n,
                                                       k,
                                                       alpha,
                                                       A,
                                                       CUDA_R_16F,
                                                       (op_A == CUBLAS_OP_N) ? m : k,
                                                       stride_A,
                                                       B,
                                                       CUDA_R_16F,
                                                       (op_B == CUBLAS_OP_N) ? k : n,
                                                       stride_B,
                                                       beta,
                                                       C,
                                                       CUDA_R_16F,
                                                       m,
                                                       stride_C,
                                                       batch,
                                                       CUDA_R_32F,
                                                       algo);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
                m,
                n,
                k,
                (int)status);
        return EXIT_FAILURE;
    }

    return 0;
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
        json j;
        std::ifstream fin(fname);
        fin >> j;
        std::vector <T> vec = j;
        
        if (vec.size() != num_elements) {
            std::cout << "the file has a tensor of different size";
            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < num_elements; i++)
            _host_data[i] = vec[i];
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
class FeedForward {
public:
    struct Config {
        int batchSize; 
        int outputSize;
        int inputSize;
        std::array<int, 3> gemm_algos;
        bool training;
        Config(int batch, int outputs, int inputs, const std::array<int, 3>& algos, bool training) 
            : batchSize(batch), outputSize(outputs), inputSize(inputs), gemm_algos(algos), training(training)
        {
        }
    };

    FeedForward(Config config) : config_(config) {}

    ~FeedForward() {}

    void ForwardCheckpoint(int bsz,  // batch * seq
                           Buffer<T>* input_ptr,
                           Buffer<T>* weights,
                           Buffer<T>* out,
                           ScheduleEngine* SE)
    {

        input_ptr->copyH2D(SE->compute);
        weights->copyH2D(SE->compute);
        out->copyH2D(SE->compute);

        cublas_fine_gemm_ex(input_ptr->get_device_data(),
                            weights->get_device_data(),
                            out->get_device_data(),
                            config_.outputSize,
                            bsz,
                            config_.inputSize,
                            SE->handle,
                            SE->compute,
                            0,
                            cublasGemmAlgo_t(config_.gemm_algos[0]));

        input_ptr->copyD2H(SE->compute);
        weights->copyD2H(SE->compute);
        out->copyD2H(SE->compute);      

    }
    /*
    void ForwardCheckpointPartitionWeights(int bsz,
                                           Buffer<T>*
                                           Buffer<T>*
                                           Buffer<T>*
                                           ScheduleEngine* SE,
                                           int nq) 
    {
        
    }
    */


    void ForwardCheckpointPartition(int bsz,  // batch * seq
                                    Buffer<T>* input_ptr,
                                    Buffer<T>* weights,
                                    Buffer<T>* out,
                                    ScheduleEngine* SE,
                                    int nq)
    {
        weights->copyH2D(SE->compute);
        int offset = 0;
        int offset_size = bsz * config_.inputSize / nq;

        #if DEBUG
            std::cout << "offset_size=" << offset_size << std::endl;
            std::cout << "input volume=" << bsz*config_.inputSize << std::endl;
            std::cout << "output volume=" << 3*bsz*config_.inputSize << std::endl;
        #endif

        for (int i = 0; i < nq; i++)
        {
            offset = i * offset_size;   
            input_ptr->copyH2D(SE->compute, offset, nq, i);
            out->copyH2D(SE->compute, offset, nq, i);
            #if DEBUG
                std::cout << "input offset=" << offset << std::endl;
                std::cout << "output offset=" << 3*offset << std::endl;
            #endif
            cublas_fine_gemm_ex(input_ptr->get_device_data(offset),
                                weights->get_device_data(),
                                out->get_device_data(3*offset),
                                config_.outputSize,
                                bsz,
                                config_.inputSize,
                                SE->handle,
                                SE->compute,
                                i,
                                cublasGemmAlgo_t(config_.gemm_algos[0]));

            input_ptr->copyD2H(SE->compute, offset, nq, i);
            out->copyD2H(SE->compute, offset, nq, i);      
        }
    }

private:
    Config config_;
};

int main(int argc, char *argv[])
{
    // PLEASE NOTE: number of queues must be less than batch_size
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int nq = atoi(argv[4]);
 
    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
    printf("Read command line parameters\n");
 
    ScheduleEngine SE(nq);
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> weights(3 * hidden_size * hidden_size, &SE);
    Buffer<float> output(3 * hidden_size * batch_size * sequence_length, &SE); 
    FeedForward<float> qkv_linear(FeedForward<float>::Config(batch_size * sequence_length, 3 * hidden_size, hidden_size, gemm_algos, true));
    qkv_linear.ForwardCheckpointPartition(batch_size * sequence_length, &input, &weights, &output, &SE, nq);
    printf("Executed qkv_linear\n");
}