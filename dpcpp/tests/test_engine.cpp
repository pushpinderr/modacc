#include <CL/sycl.hpp>
#include <stdio.h>
#include <time.h>
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
        if (device_type == 0)
            _devices = device::get_devices(info::device_type::cpu);
        else
            _devices = device::get_devices(info::device_type::gpu);

        _devices.erase(
            std::remove_if(_devices.begin(), _devices.end(),
                           [](auto Device) { return Device.is_host(); }),
            _devices.end());

        _context = context(_devices[0], &exceptionHandler);
        for (int i = 0; i < num_queues; i++)
            _queues.push_back(
                queue(_context, _devices[0], property::queue::in_order()));
    }

    void print_engine_info() {
        std::cout << "Creating " << num_queues << " queues for device "
                  << _devices[0].get_info<cl::sycl::info::device::name>()
                  << "\n";
    }
    ~ScheduleEngine() {}

    std::vector<device> _devices;
    context _context;
    int num_queues;
    std::vector<queue> _queues;
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
        printf("Asynchronous loop begins\n");

        for (int i = 0; i < se->num_queues; i++) {
            /*
                        se->_queues[i].submit([&](handler &h) {
                            h.memcpy(weights->get_device_data(i * sub_weights),
                                     weights->get_host_data(i * sub_weights),
                                     weights->get_size()/granularity);
                        });
                        */
            std::cout << "Queue " << i << " is in order? "
                      << se->_queues[i].is_in_order() << "\n";
            se->_queues[i].memcpy(weights->get_device_data(i * sub_weights),
                                  weights->get_host_data(i * sub_weights),
                                  weights->get_size() / granularity);

            auto ex = oneapi::mkl::blas::column_major::gemm(
                se->_queues[i], transA, transB, m / granularity, n, k, alpha,
                weights->get_device_data(offset * k), lda,
                input_ptr->get_device_data(), ldb, beta,
                out->get_device_data(offset * n), ldc / granularity);

            /*            se->_queues[i].submit([&](handler &h) {
                            h.memcpy(out->get_host_data(i * sub_out),
                                     out->get_device_data(i * sub_out),
                                     out->get_size() / granularity);
                        });*/
            se->_queues[i].memcpy(out->get_host_data(i * sub_out),
                                  out->get_device_data(i * sub_out),
                                  out->get_size() / granularity);

            offset += m / granularity;
        }

        for (int i = 0; i < se->num_queues; i++) {
            se->_queues[i].wait();
        }
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
        q.submit([&](handler &h) {
            h.memcpy(weights->get_device_data(), weights->get_host_data(),
                     weights->get_size());
        });

        auto ex = oneapi::mkl::blas::column_major::gemm(
            q, transA, transB, m, n, k, alpha, weights->get_device_data(), lda,
            input_ptr->get_device_data(), ldb, beta, out->get_device_data(),
            ldc);

        q.submit([&](handler &h) {
            h.memcpy(out->get_host_data(), out->get_device_data(),
                     out->get_size());
        });
        q.wait();
    }

  private:
    Config config_;
};

int main(int argc, char *argv[]) {

    Stopwatch sw;
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int qkv_size = atoi(argv[4]);
    int nq = atoi(argv[5]);
    ScheduleEngine SE(nq,0);
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
        sw.start();
        qkv_linear.ForwardPartitionWeights(&input, &weights, &output, &SE);
        sw.stop();
        printf("Time %fs\n", sw.GetTimeInSeconds());
    }
    return 0;
}
