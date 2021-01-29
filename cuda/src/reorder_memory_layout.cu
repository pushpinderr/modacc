#include "gemm.h"

#define DEV_QUERY 0
#define SEQ 1
#define ASYNC 1
#define HOST 1
#define max(a,b) (a>b)?a:b
#ifndef EVENT_PROFILE
#define EVENT_PROFILE 0
#endif

__global__ void reorder(float *in, float *out, int B, int NUM, int P)
{
        
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        int num=id/(B*P);
        int patch=id%(B*P);
        int b = patch/P;
        int p = patch%P;
        int input_ptr=num*B*P+b*P+p;
        int output_ptr=b*NUM*P+num*P+p;
        //printf("%d --> (%d %d %d) : (%d-->%d)\n",id,num,b,p,input_ptr,output_ptr);
        out[output_ptr]=in[input_ptr];
         
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
    

    int output_size = batch*sequence_length*hidden_size*3;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    size_t nbytes_C = output_size*sizeof(float);
    
    Stopwatch sw, tsync;


    // host array creation (page-locked)

    float *h_out = NULL;
    CHECK(cudaMallocHost((void **)&h_out, nbytes_C));
    initializeIncreasingValues(h_out, output_size);

    
    // device memory creation

    float *d_C = NULL;
    float *d_buf = NULL;

    CHECK(cudaMalloc((void **)&d_C, nbytes_C));
    CHECK(cudaMalloc((void **)&d_buf, nbytes_C));
    
    



#if HOST
    int total_threads,gridSize,blockSize;
    total_threads=batch*sequence_length*3*hidden_size;
    if(total_threads<1024){
        gridSize=1;
        blockSize=total_threads;
    }
    else
    {
        blockSize=1024;
        gridSize=total_threads/1024;
    }  
  //  printf("%d blocks of %d threads\n",gridSize,blockSize);
    CHECK(cudaMemcpy(d_C, h_out, nbytes_C, cudaMemcpyHostToDevice));
    sw.start();
    reorder<<<gridSize, blockSize>>>(d_C, d_buf, batch*sequence_length, num_streams,3*hidden_size/num_streams);
    sw.stop();
    double reorder_time = sw.GetTimeInSeconds();
    printf("%lf\n",reorder_time);

    CHECK(cudaMemcpy(h_out, d_buf, nbytes_C, cudaMemcpyDeviceToHost));
    //printf("GPU Reordered\n");
    //print_matrix(h_out,M,N);

#endif
   
}
