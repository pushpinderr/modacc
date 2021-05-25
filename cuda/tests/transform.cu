// %%cuda --name transform.cu
#include <bits/stdc++.h>

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
#define rows_trans 16
#define cols_trans 16

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
__global__ void Transpose_Kernel(const T* inp, T* out, int row_width, int col_width)
{
    __shared__ T data_block[rows_trans * (cols_trans + 1)];

    int r = threadIdx.x / cols_trans;
    int c = threadIdx.x % cols_trans;

    int m = row_width / cols_trans;

    int i = blockIdx.x / m * rows_trans + r;
    int j = blockIdx.x % m * cols_trans + c;

    int row_stride = rows_trans / ((rows_trans * cols_trans + THREADS - 1) / THREADS);

    for (int k = 0; k < rows_trans; k += row_stride)
        data_block[(k + r) * cols_trans + c] = inp[(i + k) * row_width + j];

    __syncthreads();

    i = blockIdx.x % m * rows_trans + r;
    j = blockIdx.x / m * cols_trans + c;

    for (int k = 0; k < rows_trans; k += row_stride)
        out[(i + k) * col_width + j] = data_block[c * cols_trans + r + k];
}

template <typename T>
void Transpose(const T* inp_mat, T* out_mat, int rows, int cols, cudaStream_t stream);

template <>
void Transpose<__half>(const __half* inp_mat,
                       __half* out_mat,
                       int rows,
                       int cols,
                       cudaStream_t stream)
{
    int threads = THREADS;

    Transpose_Kernel<__half><<<(rows * cols + threads - 1) / threads, threads, 0, stream>>>(
        inp_mat, out_mat, cols, rows);
}

template <>
void Transpose<float>(const float* inp_mat, float* out_mat, int rows, int cols, cudaStream_t stream)
{
    int threads = THREADS;

    Transpose_Kernel<float><<<(rows * cols + threads - 1) / threads, threads, 0, stream>>>(
        inp_mat, out_mat, cols, rows);
}

template <typename T>
void launch_transform_0213(T* output,
                           const T* vals,
                           int batch_size,
                           int seq_length,
                           int hidden_dim,
                           int heads,
                           cudaStream_t stream);

template <typename T>
__global__ void transform_0213(T* output,
                               const T* vals,
                               int hidden_dim,
                               int seq_length,
                               int heads,
                               int head_ext);

template <>
__global__ void transform_0213<float>(float* output,
                                      const float* vals,
                                      int hidden_dim,
                                      int seq_length,
                                      int heads,
                                      int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y / head_ext;                                       // Sequence ID (0-127)
    int d2 = threadIdx.y + (blockIdx.y % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = inputs;
}

template <>
__global__ void transform_0213<__half>(__half* output,
                                       const __half* vals,
                                       int hidden_dim,
                                       int seq_length,
                                       int heads,
                                       int head_ext)
{
#if __CUDA_ARCH__ >= 700

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y / head_ext;                                       // Sequence ID (0-127)
    int d2 = threadIdx.y + (blockIdx.y % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr[1];

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_arr[0] = vals_vec[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = vals_arr[0];
#endif
}

template <>
void launch_transform_0213<float>(float* output,
                                  const float* vals,
                                  int batch_size,
                                  int seq_length,
                                  int hidden_dim,
                                  int heads,
                                  cudaStream_t stream)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, (seq_length * head_ext));

    transform_0213<float>
        <<<grid_dim, block_dim, 0, stream>>>(output, vals, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_transform_0213<__half>(__half* output,
                                   const __half* vals,
                                   int batch_size,
                                   int seq_length,
                                   int hidden_dim,
                                   int heads,
                                   cudaStream_t stream)
{
    hidden_dim >>= 3;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, (seq_length * head_ext));
    transform_0213<__half>
        <<<grid_dim, block_dim, 0, stream>>>(output, vals, hidden_dim, seq_length, heads, head_ext);
}

// Bias add
template <typename T>
__global__ void bias_add_transform_0213(T* output,
                                        const T* vals,
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        int heads,
                                        int head_ext);

template <>
__global__ void bias_add_transform_0213<float>(float* output,
                                               const float* vals,
                                               const float* bias,
                                               int hidden_dim,
                                               int seq_length,
                                               int heads,
                                               int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride * (gridDim.z / head_ext) + cnt * d1_stride +
                             d1 * d1_stride * (gridDim.z / head_ext) + d2 * d2_stride + d3];
    float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3];

    float4 outputs;
    outputs.x = inputs.x + biases.x;
    outputs.y = inputs.y + biases.y;
    outputs.z = inputs.z + biases.z;
    outputs.w = inputs.w + biases.w;

    output_vec[cnt * d0_out_stride * gridDim.x + d0 * d0_out_stride + d1 * d1_out_stride +
               d2 * d2_out_stride + d3] = outputs;
}

#define ATTN_H 3
#define MAX_SEQ_LINE 10

template <>
__global__ void bias_add_transform_0213<__half>(__half* output,
                                                const __half* vals,
                                                const __half* bias,
                                                int hidden_dim,
                                                int seq_length,
                                                int heads,
                                                int head_ext)
{
#if __CUDA_ARCH__ >= 700

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr;
    float4 bias_arr;
    float4 output_arr;
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(&bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    bias_vec += (cnt * d1_stride);
    bias_vec += (d2 * d2_stride);

    output_vec += (cnt * d0_stride * gridDim.x);
    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    bias_arr = bias_vec[d3];
    vals_arr = vals_vec[d3];

#if defined(__ACC_HALF__)
    output_half[0] = vals_half[0] + bias_half[0];
    output_half[1] = vals_half[1] + bias_half[1];
    output_half[2] = vals_half[2] + bias_half[2];
    output_half[3] = vals_half[3] + bias_half[3];
#else
    float2 bias_arr_f[4];
    float2 vals_arr_f[4];
#pragma unroll
    for (int l = 0; l < 4; l++) {
        bias_arr_f[l] = __half22float2(bias_half[l]);
        vals_arr_f[l] = __half22float2(vals_half[l]);
        vals_arr_f[l].x += bias_arr_f[l].x;
        vals_arr_f[l].y += bias_arr_f[l].y;
        output_half[l] = __float22half2_rn(vals_arr_f[l]);
    }
#endif
    output_vec[d3] = output_arr;

#endif
}

__global__ void bias_add_transform_0213_v2(__half* output,
                                           const __half* vals,
                                           const __half* bias,
                                           int hidden_dim,
                                           int seq_length,
                                           int heads)
{
#if __CUDA_ARCH__ >= 700
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;
    int iteration_stride = d1_stride * blockDim.z;  // Hidden * 3 / 8
    int batch_stride = d0_stride * blockDim.z;      // Hidden * S * 3 / 8

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;    // Batch
    int d1 = blockIdx.y;    // Sequence ID (0-127)
    int cnt = threadIdx.z;  // blockIdx.z; // Hidden count
    int d2 = threadIdx.y;   // Head (0-11)
    int d3 = threadIdx.x;   // Values (groups of 4)

    float4 vals_arr[1];
    float4 bias_arr[1];
    float4 output_arr[1];
    __half2* vals_half = reinterpret_cast<__half2*>(vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int iter_index = cnt * d1_stride + d2 * d2_stride + d3;
    int input_offset = d0 * batch_stride + d1 * (iteration_stride << 1);
    bias_arr[0] = bias_vec[iter_index];

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        vals_arr[0] = vals_vec[input_offset + iter_id];

        output_half[0] = vals_half[0] + bias_half[0];
        output_half[1] = vals_half[1] + bias_half[1];
        output_half[2] = vals_half[2] + bias_half[2];
        output_half[3] = vals_half[3] + bias_half[3];

        in_data[iter_id] = output_arr[0];
    }
    __syncthreads();

    iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_out_stride * gridDim.x);
    int head_count = (d2 >> 1) + cnt * (blockDim.y >> 1);

    int out_index = d0 * d0_out_stride + d1 * (d1_out_stride << 1) + d3 + (d2 % 2) * d2_stride;

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = (iter * iteration_stride) + head_count;
        int iter_offset =
            (iter_row % blockDim.y) * d2_out_stride + (iter_row / blockDim.y) * matrix_stride;
        output_vec[out_index + iter_offset] =
            in_data[iter_row * d2_stride + d3 + (d2 % 2) * (d1_stride * blockDim.z)];
    }
#endif
}

template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    const T* vals,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    int hidden_dim,
                                    int heads,
                                    cudaStream_t stream,
                                    int trans_count);

// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(float* output,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           int hidden_dim,
                                           int heads,
                                           cudaStream_t stream,
                                           int trans_count)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    bias_add_transform_0213<float><<<grid_dim, block_dim, 0, stream>>>(
        output, vals, bias, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_bias_add_transform_0213<__half>(__half* output,
                                            const __half* vals,
                                            const __half* bias,
                                            int batch_size,
                                            int seq_length,
                                            int hidden_dim,
                                            int heads,
                                            cudaStream_t stream,
                                            int trans_count)
{
    hidden_dim >>= 3;
    if (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 block_dim(hidden_dim / heads, (heads / head_ext));
        dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));
        bias_add_transform_0213<__half><<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads, head_ext);
    } else {
        dim3 block_dim(hidden_dim / heads, heads, trans_count);
        dim3 grid_dim(batch_size, seq_length / 2);
        bias_add_transform_0213_v2<<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads);
    }
}

template <typename T>
__global__ void transform4d_0213(T* out,
                                 const T* in,
                                 int heads,
                                 int seq_length,
                                 int hidden_dim,
                                 int head_ext);

template <>
__global__ void transform4d_0213<float>(float* out,
                                        const float* in,
                                        int heads,
                                        int seq_length,
                                        int hidden_dim,
                                        int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = d0_stride / heads;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = hidden_dim;

    int d0 = blockIdx.x;                                        // Batch
    int d1 = blockIdx.y / ((seq_length - 1) / blockDim.y + 1);  // Head
    int d2 = (threadIdx.y + blockDim.y * blockIdx.y) % seq_length;
    int cnt = blockIdx.z;
    int d3 = threadIdx.x;  // Values (groups of 8)

    if (d2 < seq_length) {
        const float4* in_vec = reinterpret_cast<const float4*>(in);
        float4* out_vec = reinterpret_cast<float4*>(out);

        float4 vals_vec = in_vec[cnt * d0_stride * gridDim.x + d0 * d0_stride + d1 * d1_stride +
                                 d2 * d2_stride + d3];
        out_vec[d0 * d0_out_stride * gridDim.z + cnt * d2_out_stride + d1 * d1_out_stride +
                d2 * d2_out_stride * gridDim.z + d3] = vals_vec;
    }
}

template <>
__global__ void transform4d_0213<__half>(__half* out,
                                         const __half* in,
                                         int heads,
                                         int seq_length,
                                         int hidden_dim,
                                         int head_ext)
{
#if __CUDA_ARCH__ >= 700

    int d0_stride = hidden_dim * (seq_length / head_ext);
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head
    int d2 = blockIdx.z / head_ext;                                       // Sequence
    int cnt = blockIdx.y;                                                 // Hidden count
    int d3 = threadIdx.x;                                                 // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    in_vec += (cnt * d0_stride * gridDim.x);
    in_vec += (d0 * d0_stride);
    in_vec += (d2 * d2_stride);
    in_vec += (d1 * d2_stride * seq_length);

    out_vec += (cnt * d1_stride);
    out_vec += (d1 * d2_stride);
    out_vec += (d0 * d0_stride * gridDim.y);
    out_vec += (d2 * d1_stride * gridDim.y);

    out_vec[d3] = in_vec[d3];

#endif
}

__global__ void transform4d_0213_v2(__half* out,
                                    const __half* in,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim)
{
#if __CUDA_ARCH__ >= 700
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;    // Batch
    int d1 = threadIdx.y;   // Head
    int d2 = blockIdx.y;    // Sequence
    int cnt = threadIdx.z;  // Hidden count
    int d3 = threadIdx.x;   // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int input_offset = d0 * d0_stride + d2 * (d2_stride << 1) + d3 + (d1 % 2) * d2_stride;
    int head_count = (d1 >> 1) + cnt * (blockDim.y >> 1);
    int iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_stride * gridDim.x);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = iter * iteration_stride + head_count;
        int iter_offset = (iter_row % blockDim.y) * d2_stride;

        in_data[d3 + iter_offset + (iter_row / blockDim.y + (d1 % 2) * blockDim.z) * d1_stride] =
            in_vec[input_offset + iter_offset * seq_length +
                   (iter_row / blockDim.y) * matrix_stride];
    }
    __syncthreads();

    iteration_stride = d1_stride * blockDim.z;
    int iter_index = cnt * d1_stride + d1 * d2_stride + d3;
    int output_offset = d0 * d0_stride * blockDim.z + d2 * (iteration_stride << 1);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        out_vec[output_offset + iter_id] = in_data[iter_id];
    }
#endif
}
// 4D transform [0, 1, 2, 3] -> [0, 2, 1, 3]
template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             int batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             cudaStream_t stream,
                             int trans_count);

// 3 * [B A S N] - > [B S C*H]
template <>
void launch_transform4d_0213<float>(float* out,
                                    const float* in,
                                    int batch_size,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim,
                                    cudaStream_t stream,
                                    int trans_count)
{
    hidden_dim >>= 2;
    dim3 grid_dims(batch_size, heads * ((seq_length - 1) / 8 + 1), trans_count);
    dim3 block_dims(hidden_dim / heads, 8);
    transform4d_0213<float>
        <<<grid_dims, block_dims, 0, stream>>>(out, in, heads, seq_length, hidden_dim, 1);
}

template <>
void launch_transform4d_0213<__half>(__half* out,
                                     const __half* in,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     int trans_count)
{
    hidden_dim >>= 3;
    if (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 grid_dims(batch_size, trans_count, (seq_length * head_ext));
        dim3 block_dims(hidden_dim / heads, (heads / head_ext));
        transform4d_0213<__half><<<grid_dims, block_dims, 0, stream>>>(
            out, in, heads, seq_length, hidden_dim, head_ext);
    } else {
        dim3 grid_dims(batch_size, seq_length / 2);
        dim3 block_dims(hidden_dim / heads, heads, trans_count);
        transform4d_0213_v2<<<grid_dims, block_dims, 0, stream>>>(
            out, in, heads, seq_length, hidden_dim);
    }
}

int main(int argc, char* argv[]) {
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int nh = atoi(argv[4]);
    int bsz = batch_size * sequence_length;

    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "number of heads=" << nh << std::endl;

    ScheduleEngine SE(1);
    Buffer<float> attn_o_inp_ptr(bsz * sequence_length, &SE);
    Buffer<float> buf_1(bsz * sequence_length, &SE);
    launch_transform4d_0213<float> (attn_o_inp_ptr.get_device_data(), buf_1.get_device_data(), bsz, nh, sequence_length, hidden_size, SE.compute[0], 1);
    buf_1.copyD2H(SE.compute);
    buf_1.to("transform.json");
}