#pragma once
#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

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


void checkResult(float *seqRef, float *asyncRef, int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(seqRef[i] - asyncRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("sequential %5.2f asynchronous %5.2f at %d\n", seqRef[i], asyncRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}

void initializeOnes(float *input, int num_elements)
{
    for(unsigned int i = 0; i < num_elements; i++)
    {
        input[i] = 1.0;
    }
}

void print(float *input, int num_elements)
{
    for(unsigned int i = 0; i < num_elements; i++)
    {
        printf("%f ",input[i]);
    }

}

class Stopwatch {
private:
    double m_total_time;
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

            m_total_time += (double)(end_time.tv_sec - m_start_time.tv_sec) +
                            (double)(end_time.tv_nsec - m_start_time.tv_nsec) / 1e9;
        }
    }

    double GetTimeInSeconds()
    {
        if (m_is_started) {
            stop();
            start();
        }
        return m_total_time;
    }
};

