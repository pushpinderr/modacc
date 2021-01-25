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
float rand_uniform();
bool verify_result(int m, int n, int k, int ldc, float *C, float *C_reference);


int main(int argc, char*argv[])
{
    std::cout<<"GEMM Multiplication Test\n";
    Stopwatch sw;
    
    int num_streams = 4;
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
        int m = 8192;
        int k = 1024;
        int n = 3072;

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
           std::cout << "Problem size: "
                  << " A (" << m << 'x' << k << ") *"
                  << " B (" << k << 'x' << n << ")  --> "
                  << " C (" << m << 'x' << n << ")\n";
 
        std::cerr << "Launching oneMKL GEMM calculation..." << std::endl;
        
        float *h_A=(float *)malloc(sizeof(float)*m*k);
        float *d_A=(float *)malloc_device(sizeof(float)*m*k,device,context);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                h_A[i * lda + j] = rand_uniform();

        float *h_B=(float *)malloc(sizeof(float)*k*n);
        float *d_B=(float *)malloc_device(sizeof(float)*k*n,device,context);
        
        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                h_B[i * ldb + j] = rand_uniform();

        
        float *h_C=(float *)malloc(sizeof(float)*m*n);
        float *d_C=(float *)malloc_device(sizeof(float)*m*n,device,context);
        
        auto C_reference = (float *) calloc(m * n, sizeof(float));
        sw.start();
        streams[0].submit([&](handler &h) {
            h.memcpy(d_A,h_A,sizeof(float)*m*k);
        });
        streams[0].wait();
        
        streams[0].submit([&](handler &h) {
            h.memcpy(d_B,h_B,sizeof(float)*k*n);
        });
        streams[0].wait();
        
     
        oneapi::mkl::blas::row_major::gemm(streams[0], transA, transB, m, n, k,
                                           alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
        streams[0].wait_and_throw();
        
        streams[0].submit([&](handler &h) {
             h.memcpy(h_C,d_C,sizeof(float)*m*n);
        });
        streams[0].wait();
        sw.stop();
        float seq_time = sw.GetTimeInSeconds();
        printf("Time %fs\n", sw.GetTimeInSeconds());
        
        
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

float rand_uniform()
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
