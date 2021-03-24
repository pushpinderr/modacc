#include <CL/sycl.hpp>
#include <fstream>
#include <future>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <time.h>
using namespace cl::sycl;
#define NUM_ITER 3
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

void exceptionHandler(sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            std::cerr << "Caught asynchronous SYCL exception:\n"
                      << e.what() << std::endl;
        }
    }
}
class ScheduleEngine {
  public:
    ScheduleEngine(int num_queues, int device_type) {
        this->num_queues = num_queues;
        this->device_type = device_type;
        if (device_type == 0)
            _devices = device::get_devices(info::device_type::cpu);
        else if (device_type == 1)
            _devices = device::get_devices(info::device_type::gpu);

        if (device_type == 0 || device_type == 1) {
            _devices.erase(
                std::remove_if(_devices.begin(), _devices.end(),
                               [](auto Device) { return Device.is_host(); }),
                _devices.end());

            _context = context(_devices[0], &exceptionHandler);
            for (int i = 0; i < num_queues; i++)
                _queues.push_back(queue(_context, _devices[0],
                                        {property::queue::enable_profiling(),
                                         property::queue::in_order()}));
        }

        if (device_type == 2) {

            _devices_cpu = device::get_devices(info::device_type::cpu);
            _devices_cpu.erase(
                std::remove_if(_devices_cpu.begin(), _devices_cpu.end(),
                               [](auto Device) { return Device.is_host(); }),
                _devices_cpu.end());

            _context_cpu = context(_devices_cpu[0], &exceptionHandler);
            for (int i = 0; i < num_queues; i++)
                _queues_cpu.push_back(
                    queue(_context_cpu, _devices_cpu[0],
                          {property::queue::enable_profiling(),
                           property::queue::in_order()}));

            _devices = device::get_devices(info::device_type::gpu);
            _devices.erase(
                std::remove_if(_devices.begin(), _devices.end(),
                               [](auto Device) { return Device.is_host(); }),
                _devices.end());

            _context = context(_devices[0], &exceptionHandler);
            for (int i = 0; i < num_queues; i++)
                _queues.push_back(queue(_context, _devices[0],
                                        {property::queue::enable_profiling(),
                                         property::queue::in_order()}));
        }
    }

    void print_engine_info() {
        std::cout << "Creating " << num_queues << " queues for device "
                  << _devices[0].get_info<cl::sycl::info::device::name>()
                  << "\n";
    }

    void print_cpu_gpu_event_info() {
        std::ofstream myfile;
        std::string profile_mode{"partitioned_mode"};
        std::string filename =
            profile_mode + "_" + std::to_string(num_queues) + ".txt";
        myfile.open(filename.c_str());
        for (int itr = 0; itr < NUM_ITER; itr++) {
            printf("Profiling events iteration no %d\n", itr);
            auto gpu_write_start =
                writes[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_start>();

            auto gpu_write_end =
                writes[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_end>();

            auto gpu_execute_start =
                execute[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_start>();

            auto gpu_execute_end =
                execute[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_end>();

            auto gpu_read_start =
                reads[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_start>();

            auto gpu_read_end =
                reads[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_end>();

            auto cpu_execute_start =
                execute_cpu[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_start>();

            auto cpu_execute_end =
                execute_cpu[itr]
                    .get_profiling_info<
                        sycl::info::event_profiling::command_end>();

            std::cout << "==================================\n";
            std::cout << "GPU Write: " << gpu_write_start << "->"
                      << gpu_write_end << "\n";
            std::cout << "GPU Execute: " << gpu_execute_start << "->"
                      << gpu_execute_end << "\n";
            std::cout << "GPU Read: " << gpu_read_start << "->" << gpu_read_end
                      << "\n";
            std::cout << "CPU Execute: " << cpu_execute_start << "->"
                      << cpu_execute_end << "\n";

            myfile << "gpu"
                   << ":" << gpu_write_start << "," << gpu_write_end << ";"
                   << gpu_execute_start << "," << gpu_execute_end << ";"
                   << gpu_read_start << "," << gpu_read_end << "\n";
            myfile << "cpu"
                   << ":" << cpu_execute_start << "," << cpu_execute_end
                   << "\n";
        }
        myfile.close();
    }
    void print_event_info(int device, const char *dispatch_mechanism) {
        std::ofstream myfile;
        std::string filename = "queue_" + std::to_string(device) + "_" +
                               dispatch_mechanism + "_" +
                               std::to_string(num_queues) + ".txt";
        myfile.open(filename.c_str());
        std::cout << "Size of queues (write,ndrange,read)" << writes.size()
                  << " " << execute.size() << " " << reads.size() << "\n";
        for (int itr = 0; itr < NUM_ITER; itr++) {
            for (int i = 0; i < num_queues; i++) {

                auto write_start =
                    writes[itr * num_queues + i]
                        .get_profiling_info<
                            sycl::info::event_profiling::command_start>();

                auto write_end =
                    writes[itr * num_queues + i]
                        .get_profiling_info<
                            sycl::info::event_profiling::command_end>();

                auto execute_start =
                    execute[itr * num_queues + i]
                        .get_profiling_info<
                            sycl::info::event_profiling::command_start>();

                auto execute_end =
                    execute[itr * num_queues + i]
                        .get_profiling_info<
                            sycl::info::event_profiling::command_end>();

                auto read_start =
                    reads[itr * num_queues + i]
                        .get_profiling_info<
                            sycl::info::event_profiling::command_start>();

                auto read_end =
                    reads[itr * num_queues + i]
                        .get_profiling_info<
                            sycl::info::event_profiling::command_end>();
                std::cout << "==================================\n";
                std::cout << "Queue " << i << "\n";
                std::cout << "Write: " << write_start << "->" << write_end
                          << "\n";
                std::cout << "Execute: " << execute_start << "->" << execute_end
                          << "\n";
                std::cout << "Read: " << read_start << "->" << read_end << "\n";
                myfile << i << ":" << write_start << "," << write_end << ";"
                       << execute_start << "," << execute_end << ";"
                       << read_start << "," << read_end << "\n";
            }
        }

        myfile.close();
    }
    ~ScheduleEngine() {}

    std::vector<device> _devices;
    std::vector<device> _devices_cpu;
    context _context;
    context _context_cpu;
    int num_queues;
    int device_type;
    std::vector<queue> _queues;
    std::vector<queue> _queues_cpu;
    std::vector<event> writes;
    std::vector<event> reads;
    std::vector<event> execute;
    std::vector<event> execute_cpu;
};

template <typename T>

class Buffer {
  public:
    Buffer(int num_elements, ScheduleEngine *se) {
        this->num_elements = num_elements;
        auto device = se->_devices[0];
        auto context = se->_context;
        printf("Creating host data\n");
        _host_data = (float *)malloc(sizeof(T) * num_elements);
        printf("Creating device data\n");
        _device_data =
            (float *)malloc_device(sizeof(T) * num_elements, device, context);
        printf("Initializing host data\n");
        init_ones();
        printf("Finished creating Buffer\n");
    }
    T *get_host_data() { return _host_data; }
    T *get_host_data(int offset) { return _host_data + offset; }
    T *get_device_data() { return _device_data; }
    T *get_device_data(int offset) { return _device_data + offset; }

    size_t get_size() { return sizeof(T) * num_elements; }
    int get_num_elements() { return num_elements; }

    ~Buffer() {}

    void init_ones() {
        for (int i = 0; i < num_elements; i++)
            _host_data[i] = 1;
    }

    void print_host_data() {
        for (int i = 0; i < num_elements; i++)
            std::cout << _host_data[i] << "\n";
    }

  private:
    T *_host_data;
    T *_device_data;
    int num_elements;
};

template <typename T>

class FeedForward {
  public:
    struct Config {
        int batchSize, outputSize;
        int inputSize;

        Config(int batch, int inputs, int outputs)
            : batchSize(batch), outputSize(outputs), inputSize(inputs) {}
    };

    FeedForward(Config config) : config_(config) {}

    ~FeedForward() {}
    void ForwardPartitionWeightsBFS(Buffer<T> *input_ptr, Buffer<T> *weights,
                                    Buffer<T> *out, ScheduleEngine *se) {

        float alpha = T(1.);
        float beta = T(0.);
        auto transA = oneapi::mkl::transpose::trans;
        auto transB = oneapi::mkl::transpose::nontrans;
        int m = config_.outputSize;
        int k = config_.inputSize;
        int n = config_.batchSize;
        int lda = (transA == oneapi::mkl::transpose::nontrans) ? m : k;
        int ldb = (transB == oneapi::mkl::transpose::nontrans) ? k : n;
        int ldc = m;
        int granularity = se->num_queues;
        printf("Synchronous copy of input\n");
        se->_queues[0].submit([&](handler &h) {
            h.memcpy(input_ptr->get_device_data(), input_ptr->get_host_data(),
                     input_ptr->get_size());
        });
        se->_queues[0].wait();
        int sub_weights = weights->get_num_elements() / granularity;
        int sub_out = out->get_num_elements() / granularity;
        printf("Asynchronous loop begins\n");
        for (int itr = 0; itr < NUM_ITER; itr++) {

            int offset = 0;
            for (int i = 0; i < se->num_queues; i++) {
                std::cout << "Queue " << i << " is in order? "
                          << se->_queues[i].is_in_order() << "\n";
                se->writes.push_back(se->_queues[i].memcpy(
                    weights->get_device_data(i * sub_weights),
                    weights->get_host_data(i * sub_weights),
                    weights->get_size() / granularity));

                offset += m / granularity;
            }

            offset = 0;
            for (int i = 0; i < se->num_queues; i++) {
                se->execute.push_back(oneapi::mkl::blas::column_major::gemm(
                    se->_queues[i], transA, transB, m / granularity, n, k,
                    alpha, weights->get_device_data(offset * k), lda,
                    input_ptr->get_device_data(), ldb, beta,
                    out->get_device_data(offset * n), ldc / granularity));

                offset += m / granularity;
            }

            offset = 0;
            for (int i = 0; i < se->num_queues; i++) {
                se->reads.push_back(
                    se->_queues[i].memcpy(out->get_host_data(i * sub_out),
                                          out->get_device_data(i * sub_out),
                                          out->get_size() / granularity));

                offset += m / granularity;
            }

            for (int i = 0; i < se->num_queues; i++) {
                se->_queues[i].wait();
            }
        }
        se->print_event_info(se->device_type, "BFS");
    }

    void ForwardPartitionWeightsWaitAsync(Buffer<T> *input_ptr,
                                          Buffer<T> *weights, Buffer<T> *out,
                                          ScheduleEngine *se) {

        float alpha = T(1.);
        float beta = T(0.);
        auto transA = oneapi::mkl::transpose::trans;
        auto transB = oneapi::mkl::transpose::nontrans;
        int m = config_.outputSize;
        int k = config_.inputSize;
        int n = config_.batchSize;
        int lda = (transA == oneapi::mkl::transpose::nontrans) ? m : k;
        int ldb = (transB == oneapi::mkl::transpose::nontrans) ? k : n;
        int ldc = m;
        int granularity = se->num_queues;
        int offset = 0;
        printf("Synchronous copy of input\n");
        se->_queues[0].submit([&](handler &h) {
            h.memcpy(input_ptr->get_device_data(), input_ptr->get_host_data(),
                     input_ptr->get_size());
        });
        se->_queues[0].wait();
        int sub_weights = weights->get_num_elements() / granularity;
        int sub_out = out->get_num_elements() / granularity;
        Stopwatch sw;
        printf("Asynchronous loop begins\n");
        for (int itr = 0; itr < NUM_ITER; itr++) {
            sw.restart();
            offset = 0;
            std::vector<std::thread> host_waits;
            for (int i = 0; i < se->num_queues; i++) {
                std::cout << "Queue " << i << " is in order? "
                          << se->_queues[i].is_in_order() << "\n";
                se->writes.push_back(se->_queues[i].memcpy(
                    weights->get_device_data(i * sub_weights),
                    weights->get_host_data(i * sub_weights),
                    weights->get_size() / granularity));

                se->execute.push_back(oneapi::mkl::blas::column_major::gemm(
                    se->_queues[i], transA, transB, m / granularity, n, k,
                    alpha, weights->get_device_data(offset * k), lda,
                    input_ptr->get_device_data(), ldb, beta,
                    out->get_device_data(offset * n), ldc / granularity));

                /*            se->_queues[i].submit([&](handler &h) {
                                h.memcpy(out->get_host_data(i * sub_out),
                                         out->get_device_data(i * sub_out),
                                         out->get_size() / granularity);
                            });*/
                se->reads.push_back(
                    se->_queues[i].memcpy(out->get_host_data(i * sub_out),
                                          out->get_device_data(i * sub_out),
                                          out->get_size() / granularity));

                offset += m / granularity;
            }
            for (int i = 0; i < se->num_queues; i++) {

                host_waits.emplace_back(
                    std::thread([&, i]() { se->_queues[i].wait(); }));
            }
            for (auto &&t : host_waits)
                t.join();

            sw.stop();
            printf("Iteration %d: GPU MultiQueue Execution %fs\n", itr,
                   sw.GetTimeInSeconds());
        }
    se->print_event_info(se->device_type, "DFS_less_wait");
    }




    void ForwardPartitionWeights(Buffer<T> *input_ptr, Buffer<T> *weights,
                                 Buffer<T> *out, ScheduleEngine *se) {

    float alpha = T(1.);
    float beta = T(0.);
    auto transA = oneapi::mkl::transpose::trans;
    auto transB = oneapi::mkl::transpose::nontrans;
    int m = config_.outputSize;
    int k = config_.inputSize;
    int n = config_.batchSize;
    int lda = (transA == oneapi::mkl::transpose::nontrans) ? m : k;
    int ldb = (transB == oneapi::mkl::transpose::nontrans) ? k : n;
    int ldc = m;
    int granularity = se->num_queues;
    int offset = 0;
    printf("Synchronous copy of input\n");
    se->_queues[0].submit([&](handler &h) {
        h.memcpy(input_ptr->get_device_data(), input_ptr->get_host_data(),
                 input_ptr->get_size());
    });
    se->_queues[0].wait();
    int sub_weights = weights->get_num_elements() / granularity;
    int sub_out = out->get_num_elements() / granularity;
    Stopwatch sw;
    printf("Asynchronous loop begins\n");
    for (int itr = 0; itr < NUM_ITER; itr++) {
        sw.restart();
        offset = 0;
        for (int i = 0; i < se->num_queues; i++) {
           
            std::cout << "Queue " << i << " is in order? "
                      << se->_queues[i].is_in_order() << "\n";
            se->writes.push_back(
                se->_queues[i].memcpy(weights->get_device_data(i * sub_weights),
                                      weights->get_host_data(i * sub_weights),
                                      weights->get_size() / granularity));

            se->execute.push_back(oneapi::mkl::blas::column_major::gemm(
                se->_queues[i], transA, transB, m / granularity, n, k, alpha,
                weights->get_device_data(offset * k), lda,
                input_ptr->get_device_data(), ldb, beta,
                out->get_device_data(offset * n), ldc / granularity));

            se->reads.push_back(
                se->_queues[i].memcpy(out->get_host_data(i * sub_out),
                                      out->get_device_data(i * sub_out),
                                      out->get_size() / granularity));

            offset += m / granularity;
          auto ocl_q = se->_queues[i].get();
          clFlush(ocl_q);

        }
        
        sw.stop();
        printf("Iteration %d: GPU MultiQueue Execution %fs\n", itr,
               sw.GetTimeInSeconds());
    }
    for (int i = 0; i < se->num_queues; i++) {
        se->_queues[i].wait();
    }
    se->print_event_info(se->device_type, "DFS_flush_wait");
}
void ForwardPartitionWeightsAsync(Buffer<T> *input_ptr, Buffer<T> *weights,
                                  Buffer<T> *out, ScheduleEngine *se) {

    float alpha = T(1.);
    float beta = T(0.);
    auto transA = oneapi::mkl::transpose::trans;
    auto transB = oneapi::mkl::transpose::nontrans;
    int m = config_.outputSize;
    int k = config_.inputSize;
    int n = config_.batchSize;
    int lda = (transA == oneapi::mkl::transpose::nontrans) ? m : k;
    int ldb = (transB == oneapi::mkl::transpose::nontrans) ? k : n;
    int ldc = m;
    int granularity = se->num_queues;
    int offset = 0;
    printf("Synchronous copy of input\n");
    se->_queues[0].submit([&](handler &h) {
        h.memcpy(input_ptr->get_device_data(), input_ptr->get_host_data(),
                 input_ptr->get_size());
    });
    se->_queues[0].wait();
    int sub_weights = weights->get_num_elements() / granularity;
    int sub_out = out->get_num_elements() / granularity;
    Stopwatch sw;
    printf("Asynchronous loop begins\n");
    for (int itr = 0; itr < NUM_ITER; itr++) {
        sw.restart();
        offset = 0;
        std::vector<std::thread> async_threads;
        for (int i = 0; i < se->num_queues; i++) {
            async_threads.emplace_back(std::thread([&, i, offset]() {
                //                    std::cout<<"Thread "<<i<<" begins\n";
                se->_queues[i].memcpy(weights->get_device_data(i * sub_weights),
                                      weights->get_host_data(i * sub_weights),
                                      weights->get_size() / granularity);

                oneapi::mkl::blas::column_major::gemm(
                    se->_queues[i], transA, transB, m / granularity, n, k,
                    alpha, weights->get_device_data(offset * k), lda,
                    input_ptr->get_device_data(), ldb, beta,
                    out->get_device_data(offset * n), ldc / granularity);

                se->_queues[i].memcpy(out->get_host_data(i * sub_out),
                                      out->get_device_data(i * sub_out),
                                      out->get_size() / granularity);
                //            std::cout<<"Thread "<<i<<" ends\n";
                se->_queues[i].wait();
            }));

            offset += m / granularity;
        }
        printf("Number of async threads: %zu\n", async_threads.size());
        for (auto &&t : async_threads)
            t.join();
        sw.stop();
        printf("Iteration %d: GPU MultiQueue Async Execution %fs\n", itr,
               sw.GetTimeInSeconds());
    }
    //         se->print_event_info(se->device_type, "DFS_Async");
}

void ForwardPartitionWeights_CPU_GPU(Buffer<T> *input_ptr, Buffer<T> *weights,
                                     Buffer<T> *out, ScheduleEngine *se,
                                     int pr) {

    Stopwatch sw;
    float alpha = T(1.);
    float beta = T(0.);
    auto transA = oneapi::mkl::transpose::trans;
    auto transB = oneapi::mkl::transpose::nontrans;
    int m = config_.outputSize;
    int k = config_.inputSize;
    int n = config_.batchSize;
    int lda = (transA == oneapi::mkl::transpose::nontrans) ? m : k;
    int ldb = (transB == oneapi::mkl::transpose::nontrans) ? k : n;
    int ldc = m;
    int granularity;
    int offset = 0;
    printf("Synchronous copy of input\n");
    se->_queues[0].submit([&](handler &h) {
        h.memcpy(input_ptr->get_device_data(), input_ptr->get_host_data(),
                 input_ptr->get_size());
    });
    se->_queues[0].wait();
    int sub_weights = weights->get_num_elements() / granularity;
    int sub_out = out->get_num_elements() / granularity;
    printf("Asynchronous loop begins\n");

    for (int itr = 0; itr < NUM_ITER; itr++) {
        sw.restart();
        offset = 0;
        int i = 0;
        granularity = pr;
        // GPU Execution

        se->writes.push_back(
            se->_queues[0].memcpy(weights->get_device_data(i * sub_weights),
                                  weights->get_host_data(i * sub_weights),
                                  weights->get_size() / granularity));

        se->execute.push_back(oneapi::mkl::blas::column_major::gemm(
            se->_queues[0], transA, transB, m / granularity, n, k, alpha,
            weights->get_device_data(offset * k), lda,
            input_ptr->get_device_data(), ldb, beta,
            out->get_device_data(offset * n), ldc / granularity));

        se->reads.push_back(se->_queues[0].memcpy(
            out->get_host_data(i * sub_out), out->get_device_data(i * sub_out),
            out->get_size() / granularity));
        offset += m / granularity;

        // CPU Execution
        auto async_cpu_thread =
            std::async([&se, &weights, &input_ptr, &out, lda, ldb, ldc, m, n, k,
                        alpha, beta, transA, transB, granularity, offset]() {
                se->execute_cpu.push_back(oneapi::mkl::blas::column_major::gemm(
                    se->_queues_cpu[0], transA, transB, m - m / granularity, n,
                    k, alpha, weights->get_host_data(offset * k), lda,
                    input_ptr->get_host_data(), ldb, beta,
                    out->get_host_data(offset * n), ldc - ldc / granularity));

                se->_queues_cpu[0].wait();
            });

        se->_queues[0].wait();
        async_cpu_thread.wait();
        sw.stop();
        printf("Iteration %d: CPU/GPU Partitioned Execution %fs\n", itr,
               sw.GetTimeInSeconds());
    }
    se->print_cpu_gpu_event_info();
}

void Forward(Buffer<T> *input_ptr, Buffer<T> *weights, Buffer<T> *out,
             ScheduleEngine *se) {
    float alpha = T(1.);
    float beta = T(0.);
    auto transA = oneapi::mkl::transpose::trans;
    auto transB = oneapi::mkl::transpose::nontrans;
    int m = config_.outputSize;
    int k = config_.inputSize;
    int n = config_.batchSize;
    int lda = (transA == oneapi::mkl::transpose::nontrans) ? m : k;
    int ldb = (transB == oneapi::mkl::transpose::nontrans) ? k : n;
    int ldc = m;
    auto q = se->_queues[0];
    q.submit([&](handler &h) {
        h.memcpy(input_ptr->get_device_data(), input_ptr->get_host_data(),
                 input_ptr->get_size());
    });
    Stopwatch sw;
    for (int itr = 0; itr < NUM_ITER; itr++) {
        sw.restart();
        se->writes.push_back(q.submit([&](handler &h) {
            h.memcpy(weights->get_device_data(), weights->get_host_data(),
                     weights->get_size());
        }));

        se->execute.push_back(oneapi::mkl::blas::column_major::gemm(
            q, transA, transB, m, n, k, alpha, weights->get_device_data(), lda,
            input_ptr->get_device_data(), ldb, beta, out->get_device_data(),
            ldc));

        se->reads.push_back(q.submit([&](handler &h) {
            h.memcpy(out->get_host_data(), out->get_device_data(),
                     out->get_size());
        }));
        q.wait();
        sw.stop();
        printf("Iteration %d: Single Device Execution %fs\n", itr,
               sw.GetTimeInSeconds());
    }
    se->print_event_info(se->device_type, "NoPart");
}

private:
Config config_;
}
;

int main(int argc, char *argv[]) {

    Stopwatch sw;
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int qkv_size = atoi(argv[4]);
    int nq = atoi(argv[5]);
    int pr = nq;
    int device_type = atoi(argv[6]);
    ScheduleEngine SE(nq, device_type);
    SE.print_engine_info();

    printf("Creating Input\n");
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    printf("Creating Weights\n");
    Buffer<float> weights(hidden_size * qkv_size, &SE);
    printf("Creating Output\n");
    Buffer<float> output(batch_size * sequence_length * qkv_size, &SE);
    printf("Creating FeedForward Instance\n");
    FeedForward<float> qkv_linear(FeedForward<float>::Config(
        batch_size * sequence_length, hidden_size, qkv_size));
    if (nq == 1) {
        sw.start();
        qkv_linear.Forward(&input, &weights, &output, &SE);
        sw.stop();
        printf("Time %fs\n", sw.GetTimeInSeconds());

    } else {
        if (device_type < 2) {
            sw.start();

            qkv_linear.ForwardPartitionWeights(&input, &weights,
                                                        &output, &SE);
            sw.stop();
            printf("Time %fs\n", sw.GetTimeInSeconds());
        } else {

            sw.start();
            qkv_linear.ForwardPartitionWeights_CPU_GPU(&input, &weights,
                                                       &output, &SE, pr);
            sw.stop();
            printf("Time %fs\n", sw.GetTimeInSeconds());
        }
    }
    return 0;
}