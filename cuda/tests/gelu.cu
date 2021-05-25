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

namespace cg = cooperative_groups;

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

inline __device__ float gelu(const float x) // gelu activation function.
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
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
class Gelu {
public:
    struct Config {
        uint32_t intermediate_size;
        Config(uint32_t inter_size) : intermediate_size(inter_size) {}
    };

    Gelu(const Config& config) : _config(config) {}

    ~Gelu() {}

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


int main(int argc, char *argv[])
{
    // PLEASE NOTE: number of queues must be less than batch_size
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int intermediate_size = atoi(argv[3]);
    int nq = atoi(argv[4]);
    int bsz = sequence_length * batch_size;

    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "intermediate size=" << intermediate_size << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
 
    ScheduleEngine SE(nq);
    Buffer<float> input(bsz * sequence_length, &SE);
    Buffer<float> bias(batch_size * sequence_length, &SE);
    Buffer<float> output(batch_size * sequence_length, &SE);
    // Gelu<float> GeLU(Gelu<float>::Config(intermediate_size));
    Gelu<float> GELU(Gelu<float>::Config(intermediate_size));

    for ( int i = 0; i < nq; i++ ) 
    {
        
    }
    GELU.ForwardWithBiasAdd(bsz, &input, &bias, &output, &SE);
    CHECK(cudaThreadSynchronize());
    printf("Executed Gelu\n");
}
