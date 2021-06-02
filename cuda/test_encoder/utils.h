#pragma once

#include "cuda.h"
#include <vector>
#include <cassert>
#include "curand.h"
#include <assert.h>
#include <iostream>
#include "cublas_v2.h"
#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#ifndef EVENT_PROFILE
#define EVENT_PROFILE 0
#endif



#define DEBUG false // flag for debugging.
#define THREADS 256
#define TILE_DIM 32
#define WARP_SIZE 32
#define NUM_STREAMS 32
#define MAX_THREADS 1024
#define MAX_WARP_NUM 32
#define MAX_REGISTERS 256
#define MAX_THREAD_STRIDE 32
#define MAX_THREAD_ITERATIONS 8  // Maximum 8K.
#define NORM_REG (MAX_REGISTERS / 4)

const int unroll_factor = 4; // unrolling factor.

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

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

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                                          \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x) \
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); j += blockDim.y * gridDim.y)

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return (std::max)(
        (std::min)((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block.
        1);
}

void fileWrite(std::string fname, std::string str) {
    std::ofstream fout(fname);
    fout << str;
    fout.close();
}

std::pair <uint64_t, uint64_t> getSeed(uint64_t increment, uint64_t seed) {
    return std::pair<uint64_t, uint64_t>(seed, increment);
}

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
        #if DEBUG
            printf("Creating host data\n");
        #endif
        CHECK(cudaMallocHost((void **)&_host_data, sizeof(T)*num_elements));
        #if DEBUG
            printf("Creating device data\n");
        #endif
        CHECK(cudaMalloc((void **)&_device_data, sizeof(T)*num_elements));
        #if DEBUG
            printf("Initializing host data\n");
        #endif
        init_ones();
        #if DEBUG
            printf("Finished creating Buffer\n");
        #endif
    }
    T *get_host_data() { return _host_data; }
    T *get_host_data(int offset) { return _host_data + offset; }
    T *get_device_data() { return _device_data; }
    T *get_device_data(int offset) { return _device_data + offset; }

    size_t get_size() { return sizeof(T) * num_elements; }
    size_t get_size(int nq) { return sizeof(T) * (num_elements/nq); }
    int get_num_elements() { return num_elements; }

    ~Buffer() {}

    void init(int val=1) {
        for (int i = 0; i < num_elements; i++)
            _host_data[i] = val;        
    }

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

    void binarize(int thresh=0.5) {
        for ( int i = 0; i < num_elements; i++ ) {
            _host_data[i] = (( _host_data[i] > thresh ) ? 1 : 0);
        }
    }

    void print_host_data(int start=0, int end=-1) {
        if ( start < 0 ) 
            start = num_elements - start;
        if ( end < 0 ) 
            end = num_elements - end;

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
