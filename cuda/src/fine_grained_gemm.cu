#include "gemm.h"

#define DEV_QUERY 0
#define SEQ 1
#define ASYNC 1
#define max(a,b) (a>b)?a:b
#ifndef EVENT_PROFILE
#define EVENT_PROFILE 0
#endif


void gpu_gemm(cublasHandle_t handle, struct GemmConfig *config, float *weight, float *input, float *output, int granularity, int offset)
{

    //offset refers to the offset via number of sub-rows dicated by granularity

    float *d_A = weight + offset * config->k;
    float *d_B = input; 
    float *d_C = output + offset * config->n;
    
    long long int lda = config->lda;
    long long int ldb = config->ldb;
    long long int ldc = config->ldc / granularity;
    

    CHECK_CUBLAS(cublasSgemm(handle, config->op_A, config->op_B, config->m/granularity, config->n, config->k, &config->alpha, d_A, lda, d_B, ldb, &config->beta, d_C, ldc));


}

int main(int argc, char *argv[])
{
    //Query Device
#if DEV_QUERY
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting...\n", argv[0]);
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // check if device supports hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
            deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

#endif
    //Variable Initialization
    
    
    
    
    long long int batch = atoi(argv[1]);
    long long int sequence_length = atoi(argv[2]);
    long long int hidden_size = atoi(argv[3]); //Note here hidden_size is hidden_size per Q,K,V 
    int num_streams = atoi(argv[4]);
    

    int input_size = batch*sequence_length*hidden_size;
    int weight_size = hidden_size*hidden_size*3;
    int output_size = batch*sequence_length*hidden_size*3;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    size_t nbytes_A = weight_size*sizeof(float);
    size_t nbytes_B = input_size*sizeof(float);
    size_t nbytes_C = output_size*sizeof(float);
    
    Stopwatch sw, tsync;


    // host array creation (page-locked)

    float *h_input = NULL; 
    float *h_weight = NULL;
    float *h_out = NULL;
    float *ref = NULL;
    CHECK(cudaMallocHost((void **)&h_input, nbytes_B));
    CHECK(cudaMallocHost((void **)&h_weight, nbytes_A));
    CHECK(cudaMallocHost((void **)&h_out, nbytes_C));
    CHECK(cudaMallocHost((void **)&ref, nbytes_C));

    initializeOnes(h_input, input_size);
    initializeOnes(h_weight, weight_size);
    initializeOnes(h_out, output_size);

    
    // device memory creation

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    CHECK(cudaMalloc((void **)&d_A, nbytes_A));
    CHECK(cudaMalloc((void **)&d_B, nbytes_B));
    CHECK(cudaMalloc((void **)&d_C, nbytes_C));


    // stream configuration


    cudaStream_t compute[num_streams];

    cudaEvent_t start[num_streams], stop[num_streams];
    cudaEvent_t seq_start, seq_end;
    CHECK(cudaEventCreate(&seq_start));
    CHECK(cudaEventCreate(&seq_end));
    for(int i=0;i<num_streams;i++)
    {
        CHECK(cudaEventCreate(&start[i]));
        CHECK(cudaEventCreate(&stop[i]));
    }


    for(int i=0;i<num_streams;i++)
    {
        CHECK(cudaStreamCreate(&compute[i]));
    }


    cublasOperation_t op_A = CUBLAS_OP_T;
    cublasOperation_t op_B = CUBLAS_OP_N;


    GemmConfig config(op_A,op_B,batch,sequence_length,hidden_size,1.0,0);

    //Sequential Operations

#if SEQ

    sw.start();
    //H2D Copy

    CHECK(cudaMemcpy(d_B, h_input, nbytes_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_A, h_weight, nbytes_A, cudaMemcpyHostToDevice));
    
   //execute
   #if EVENT_PROFILE 
   cudaEventRecord(seq_start);
   #endif
    gpu_gemm(handle, &config, d_A, d_B, d_C, 1, 0);
   #if EVENT_PROFILE
    cudaEventRecord(seq_end);
     
   #endif

    // D2H Copy

    CHECK(cudaMemcpy(ref, d_C, nbytes_C, cudaMemcpyDeviceToHost));
    sw.stop();
    double seq_time = sw.GetTimeInSeconds();
    // printf("Sequential Time %lfs\n", sw.GetTimeInSeconds());
    sw.restart();
    #if EVENT_PROFILE
    cudaEventSynchronize(seq_end);
    float event_recorded_time = 0;
    cudaEventElapsedTime(&event_recorded_time, seq_start, seq_end);
    //printf("default: %lf\n",event_recorded_time);
    #endif
#endif


#if ASYNC

// Multistream implementation 

    int granularity = num_streams;
    int offset = 0;
    int buffer_offset_A, buffer_offset_C;
    int sub_A =  weight_size/granularity;    
    size_t sub_nbytes_A = nbytes_A/granularity;
    int sub_C =  output_size/granularity;       
    size_t sub_nbytes_C = nbytes_C/granularity;

    sw.start();
    CHECK(cudaMemcpy(d_B, h_input, nbytes_B, cudaMemcpyHostToDevice));
    for(int i=0;i<num_streams;i++)
    {
        buffer_offset_A = i*sub_A;    
        CHECK(cudaMemcpyAsync(&d_A[buffer_offset_A], &h_weight[buffer_offset_A], sub_nbytes_A,
                              cudaMemcpyHostToDevice, compute[i]));
        
        cublasSetStream(handle,compute[i]);
        #if EVENT_PROFILE 
        cudaEventRecord(start[i],compute[i]);
        #endif
        gpu_gemm(handle, &config, d_A, d_B, d_C, granularity,offset);
        #if EVENT_PROFILE
        cudaEventRecord(stop[i],compute[i]);
        #endif
        buffer_offset_C = i*sub_C;    
        CHECK(cudaMemcpyAsync(&h_out[buffer_offset_C], &d_C[buffer_offset_C], sub_nbytes_C,
                              cudaMemcpyDeviceToHost, compute[i]));
        offset += 3*hidden_size/granularity;
    }
    
    CHECK(cudaThreadSynchronize());    
    sw.stop();
    double async_time = sw.GetTimeInSeconds();
    sw.restart();
#endif
#if EVENT_PROFILE
for(int i=0;i<num_streams;i++)
{
    cudaEventSynchronize(stop[i]);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start[i], stop[i]);
    printf("stream %d: %lf\n",i,milliseconds);
}
#endif

#if EVENT_PROFILE==0
    printf("%lf %lf\n",seq_time,async_time);
#endif
   
}
