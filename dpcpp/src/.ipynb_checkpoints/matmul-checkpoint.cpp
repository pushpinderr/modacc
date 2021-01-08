//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Matrix Multiplication is a simple program that multiplies together two
//     large matrices and verifies the results.
// This samples uses the oneAPI Math Kernel Library (oneMKL) to accelerate
//     the computation.

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>
using namespace cl::sycl;

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

float rand_uniform();
bool verify_result(int m, int n, int k, int ldc, float *C, float *C_reference);


int main(int argc, char*argv[])
{
    std::cout<<"GEMM Multiplication Test\n";
    
    int num_streams = 4;
    gpu_selector selector;
    queue streams[num_streams];
    for(int i=0;i<num_streams;i++)
        streams[i]=queue(selector);
    
//     queue h2d(selector);
//     queue d2h(selector);
//     queue execute(selector);
    for(int i=0;i<num_streams;i++)
       std::cout << "Stream " << i<<" "<<streams[i].get_device().get_info<info::device::name>() << "\n";
    
//     std::cout << "Device for H2D: " << h2d.get_device().get_info<info::device::name>() << std::endl;
//     std::cout << "Device for D2H: " << d2h.get_device().get_info<info::device::name>() << std::endl;
//     std::cout << "Device for Execute: " << execute.get_device().get_info<info::device::name>() << std::endl;
   
    try {
        // Initialize data for GEMM. The full GEMM operation is:
        //
        //      C = alpha * op(A) * op(B) + beta * C
        //
        // where alpha, beta are scalar values, and op(...) represents
        // optional matrix transposition.
        //
        // For this simple matrix multiplication, no transposition is needed.
        // 
        // By choosing alpha = 1, beta = 0, GEMM will calculate C = A * B.
        //
        // In this example, matrices are stored in row-major layout.

        auto transA = oneapi::mkl::transpose::nontrans;
        auto transB = oneapi::mkl::transpose::nontrans;

        // Matrix data sizes.
        // 
        // A is m x k
        // B is k x n  --> product C is m x n
        int m = 60;
        int k = 120;
        int n = 240;

        // Leading dimensions of data. For row-major matrices, the leading
        // dimension is the stride between adjacent rows.
        int lda = k;
        int ldb = n;
        int ldc = n;

        // Scaling factors.
        double alpha = 1.0;
        double beta = 0.0;

        // Create a queue on the default device.
//         sycl::queue device_queue{sycl::default_selector{}};

//         std::cout << "Device: "
//                   << device_queue.get_device().get_info<sycl::info::device::name>()
//                   << std::endl;
        
        auto device = streams[0].get_device();
        auto context = streams[0].get_context();
        
        // Allocate host and device memory for matrices
        
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
        
        streams[0].submit([&](handler &h) {
            h.memcpy(d_A,h_A,sizeof(float)*m*k);
        });
        streams[0].wait();
        
        streams[0].submit([&](handler &h) {
            h.memcpy(d_B,h_B,sizeof(float)*k*n);
        });
        streams[0].wait();
        
        std::cout << "Problem size: "
                  << " A (" << m << 'x' << k << ") *"
                  << " B (" << k << 'x' << n << ")  --> "
                  << " C (" << m << 'x' << n << ")\n";
 
        std::cerr << "Launching oneMKL GEMM calculation..." << std::endl;
        oneapi::mkl::blas::row_major::gemm(streams[0], transA, transB, m, n, k,
                                           alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
        streams[0].wait_and_throw();
        
        streams[0].submit([&](handler &h) {
             h.memcpy(h_C,d_C,sizeof(float)*m*n);
        });
        streams[0].wait();
        
        
        std::cerr << "Performing reference calculation..." << std::endl;
        for (int i = 0; i < m; i++)
            for (int h = 0; h < k; h++)
                for (int j = 0; j < n; j++)
                    C_reference[i * ldc + j] += h_A[i * lda + h] * h_B[h * ldb + j];
        
        bool ok = verify_result(m, n, k, ldc, h_C, C_reference);
        std::cerr<< "Computation OK? "<<ok<<"\n";
        
//         streams[0].submit([&](handler &h) {
//              h.parallel_for(range<1>(m*k), [=](id<1> i) {
//                 d_A[i]=2;
//               });
//         });
//         streams[0].wait();
        
        
//         streams[0].submit([&](handler &h) {
//              h.memcpy(h_A,d_A,sizeof(float)*m*k);
//         });
//         streams[0].wait();
        
//         for (int i = 0; i < m*k; i++)
//             std::cout<<h_A[i]<<"\n";
        
        /*
        // Allocate shared memory for matrices.
        auto A = sycl::malloc_shared<double>(m * k, device_queue);
        auto B = sycl::malloc_shared<double>(k * n, device_queue);
        auto C = sycl::malloc_shared<double>(m * n, device_queue);
        auto C_reference = (double *) calloc(m * n, sizeof(double));

        if (!A || !B || !C || !C_reference) {
            std::cerr << "Could not allocate memory for matrices." << std::endl;
            exit(1);
        }

        // Initialize matrix data.
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                A[i * lda + j] = rand_uniform();

        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                B[i * ldb + j] = rand_uniform();

        std::cout << "Problem size: "
                  << " A (" << m << 'x' << k << ") *"
                  << " B (" << k << 'x' << n << ")  --> "
                  << " C (" << m << 'x' << n << ")\n";

        // Call GEMM to do matrix multiplication, asynchronously.
        std::cerr << "Launching oneMKL GEMM calculation..." << std::endl;
        oneapi::mkl::blas::row_major::gemm(device_queue, transA, transB, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);

        // While calculation occurs, compute reference result to check accuracy.
        std::cerr << "Performing reference calculation..." << std::endl;
        for (int i = 0; i < m; i++)
            for (int h = 0; h < k; h++)
                for (int j = 0; j < n; j++)
                    C_reference[i * ldc + j] += A[i * lda + h] * B[h * ldb + j];
        
        // Wait for oneMKL computation to complete.
        device_queue.wait_and_throw();

        // Check results for accuracy.
        bool ok = verify_result(m, n, k, ldc, C, C_reference);

        // Free memory.
        free(A, device_queue);
        free(B, device_queue);
        free(C, device_queue);
        free(C_reference);

        if (!ok)
            exit(2);
*/
    } catch (const std::exception &e) {
            std::cerr << "An exception occurred: "
                      << e.what() << std::endl;
            exit(1);
        }

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
