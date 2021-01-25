
#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
#include<ctime>
using namespace cl::sycl;

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif
class Stopwatch {
private:
    float m_total_time;
    struct timespec m_start_time;
    bool m_is_started;

public:
    Stopwatch()
    {
        m_total_time = 0.0;
        m_is_started = false;
    }

    ~Stopwatch() {}

    void Reset() { m_total_time = 0.0; }

    void start()
    {
        clock_gettime(CLOCK_MONOTONIC, &m_start_time);
        m_is_started = true;
    }

    void restart()
    {
        m_total_time = 0.0;
        clock_gettime(CLOCK_MONOTONIC, &m_start_time);
        m_is_started = true;
    }

    void stop()
    {
        if (m_is_started) {
            m_is_started = false;

            struct timespec end_time;
            clock_gettime(CLOCK_MONOTONIC, &end_time);

            m_total_time += (float)(end_time.tv_sec - m_start_time.tv_sec) +
                            (float)(end_time.tv_nsec - m_start_time.tv_nsec) / 1e9;
        }
    }

    float GetTimeInSeconds()
    {
        if (m_is_started) {
            stop();
            start();
        }
        return m_total_time;
    }
};
float init_ones();
bool verify_result(int m, int n, int k, int ldc, float *C, float *C_reference);


int main(int argc, char*argv[])
{
    std::cout<<"GEMM Multiplication Test\n";
    Stopwatch sw;
    int batch = 8;
    int sequence_length = 1024;
    int hidden_size = 1024; //Note here hidden_size is hidden_size per Q,K,V 
    int num_streams = 8;
    int event_flag=0;
    gpu_selector selector;
    queue streams[num_streams];
    for(int i=0;i<num_streams;i++)
        streams[i]=queue(selector);
    
    
    for(int i=0;i<num_streams;i++)
       std::cout << "Stream " << i<<" "<<streams[i].get_device().get_info<info::device::name>() << "\n";
    
    //Coarse Grained

    try {
     

        auto transA = oneapi::mkl::transpose::nontrans;
        auto transB = oneapi::mkl::transpose::nontrans;

        // Matrix data sizes.
        // 
        // A is m x k
        // B is k x n  --> product C is m x n
        int m = batch*sequence_length;
        int k = hidden_size;
        int n = 3*hidden_size;

        // Leading dimensions of data. For row-major matrices, the leading
        // dimension is the stride between adjacent rows.
        int lda = k;
        int ldb = n;
        int ldc = n;

        // Scaling factors.
        double alpha = 1.0;
        double beta = 0.0;


        
        auto device = streams[0].get_device();
        auto context = streams[0].get_context();
        
        // Allocate host and device memory for matrices
        
        float *h_A=(float *)malloc(sizeof(float)*m*k);
        float *d_A=(float *)malloc_device(sizeof(float)*m*k,device,context);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                h_A[i * lda + j] = init_ones();

        float *h_B=(float *)malloc(sizeof(float)*k*n);
        float *d_B=(float *)malloc_device(sizeof(float)*k*n,device,context);
        
        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                h_B[i * ldb + j] = init_ones();

        
        float *h_C=(float *)malloc(sizeof(float)*m*n);
        float *d_C=(float *)malloc_device(sizeof(float)*m*n,device,context);
        
        auto C_reference = (float *) calloc(m * n, sizeof(float));
        
        // Variable for partitioning matrices A and C
        int input_size = batch*sequence_length*hidden_size;
        int weight_size = hidden_size*hidden_size*3;
        int output_size = batch*sequence_length*hidden_size*3;

        std::cout<<"Input : "<<input_size<<"\n";
        std::cout<<"Weight : "<<weight_size<<"\n";
        std::cout<<"Output : "<<weight_size<<"\n";
        
        size_t nbytes_A = input_size*sizeof(float);
        size_t nbytes_B = weight_size*sizeof(float);
        size_t nbytes_C = output_size*sizeof(float);
        
        std::cout<<"Input Size: "<<nbytes_A<<"\n";
        std::cout<<"Weight Size: "<<nbytes_B<<"\n";
        std::cout<<"Output Size: "<<nbytes_C<<"\n";
        
        
        int granularity = num_streams;
        int offset = 0;
        int buffer_offset_A, buffer_offset_C;
        int sub_A =  input_size/granularity;    
        size_t sub_nbytes_A = nbytes_A/granularity;
        int sub_C =  output_size/granularity;       
        size_t sub_nbytes_C = nbytes_C/granularity;
        
        std::cout << "Problem size: "
                  << " A (" << m << 'x' << k << ") *"
                  << " B (" << k << 'x' << n << ")  --> "
                  << " C (" << m << 'x' << n << ")\n";
 
        std::cerr << "Launching oneMKL GEMM calculation using fine-grained scheduling" << std::endl;
        sw.start();
        streams[0].submit([&](handler &h) {
            h.memcpy(d_B,h_B,sizeof(float)*k*n);
        });

        streams[0].wait();
        for(int i=0;i<num_streams;i++)
        {
            std::cerr << "Initiating stream " << i <<std::endl;
            buffer_offset_A = i*sub_A;
            std::cerr << "A offset,starting point,size "<<i<<","<<buffer_offset_A<<","<<sub_nbytes_A<<"\n";
            event h2d,ex;
            if(event_flag)
                h2d=streams[i].submit([&](handler &h) {
                    h.memcpy(&d_A[buffer_offset_A],&h_A[buffer_offset_A],sub_nbytes_A);
                });
            else
            {
                streams[i].submit([&](handler &h) {
                    h.memcpy(&d_A[buffer_offset_A],&h_A[buffer_offset_A],sub_nbytes_A);
                });
                streams[i].submit_barrier();
            }
            
            std::cerr << "Launching GEMM instance "<<i<<"\n";
            if(event_flag)
                ex=oneapi::mkl::blas::row_major::gemm(streams[i], transA, transB, m/granularity, n, k,
                                               alpha, d_A+offset*k, lda, d_B, ldb, beta, d_C+offset*n, ldc,{h2d});
            else
            {
                oneapi::mkl::blas::row_major::gemm(streams[i], transA, transB, m/granularity, n, k,
                                               alpha, d_A+offset*k, lda, d_B, ldb, beta, d_C+offset*n, ldc);
                streams[i].submit_barrier();
            }
            buffer_offset_C = i*sub_C;   
            
            std::cerr << "C offset,starting point,size: "<<i<<","<<buffer_offset_C<<","<<sub_nbytes_C<<"\n";
            if(event_flag)
                streams[i].submit([&](handler &h) {
                     h.depends_on(ex);
                     h.memcpy(&h_C[buffer_offset_C],&d_C[buffer_offset_C],sub_nbytes_C);
                });
            else
            {
                streams[i].submit([&](handler &h) {
                   h.memcpy(&h_C[buffer_offset_C],&d_C[buffer_offset_C],sub_nbytes_C);
                });
                
            }

            offset=offset+m/granularity;
        }

        for(int i=0;i<num_streams;i++)
            streams[i].wait();
         sw.stop();
         printf("Time %fs\n", sw.GetTimeInSeconds());
        
//         std::cout<<"Reference\n";
// //         ldc=n;
//         for (int i = 0; i < m; i++)
//             for (int h = 0; h < k; h++)
//                 for (int j = 0; j < n; j++)
//                     C_reference[i * ldc + j] += h_A[i * lda + h] * h_B[h * ldb + j];

//         for (int i = 0; i < m; i++)
//         {
//            for (int j = 0; j < n; j++)
//               std::cout<<C_reference[i * ldc + j] <<" ";
//             std::cout<<"\n";
//         }
        
//         std::cout<<"Computed\n";
//         for (int i = 0; i < m; i++)
//         {
//            for (int j = 0; j < n; j++)
//               std::cout<<h_C[i * ldc + j] <<" ";
//             std::cout<<"\n";
//         }
        
//         std::cerr << "Performing reference calculation..." << std::endl;
//         for (int i = 0; i < m; i++)
//             for (int h = 0; h < k; h++)
//                 for (int j = 0; j < n; j++)
//                     C_reference[i * ldc + j] += h_A[i * lda + h] * h_B[h * ldb + j];
        
//         bool ok = verify_result(m, n, k, ldc, h_C, C_reference);
//         std::cerr<< "Computation OK? "<<ok<<"\n";
        


    } catch (const std::exception &e) {
            std::cerr << "An exception occurred: "
                      << e.what() << std::endl;
            exit(1);
        }

    //Fine Grained
    
    
    
}

float init_ones()
{
    return 1.0; 
}

bool verify_result(int m, int n, int k, int ldc, float *C, float *C_reference)
{
    double tolerance = 1e-6;
    bool ok = true;

    // Compare host side results with the result buffer from device side: print
    // fail data 5 times only.
    int printf_count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto idx = i * ldc + j;
            auto abs_diff = std::abs(C[idx] - C_reference[idx]);

            if (abs_diff > tolerance && printf_count++ < 5) {
                std::cerr << "The result is incorrect for element "
                          << '[' << i << ", " << j << ']'
                          << ", expected: " << C_reference[idx]
                          << ", but got: " << C[idx] << std::endl;
                ok = false;
            }
        }
    }

    if (ok)
        std::cout << "Results are accurate.\n";

    return ok;
    
}
