#include <CL/sycl.hpp>
#include <stdio.h>
using namespace cl::sycl;

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif
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
    ScheduleEngine(int num_queues) {
        this->num_queues = num_queues;
        _devices = device::get_devices(info::device_type::cpu);

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
        T *_host_data = (float *)malloc(sizeof(T) * num_elements);
        printf("Creating device data\n");
        T *_device_data =
            (float *)malloc_device(sizeof(T) * num_elements, device, context);
        printf("Initializing host data\n");
        init_ones();
        printf("Finished creating Buffer\n");
    }
    T *get_host_data() { return _host_data; }
    T *get_device_data() { return _device_data; }
    size_t get_size() { return sizeof(T) * num_elements; }

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
    /*
        void ForwardPartitionWeights(int bsz, const T *input_ptr, const T
       *weights, T *out, ScheduleEngine *se) {

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
            for (int i = 0; i < se->num_queues; i++) {
                auto ex = oneapi::mkl::blas::row_major::gemm(
                    se->_queues[i], transA, transB, m / granularity, n, k,
       alpha, weights + offset * k, lda, input_ptr, ldb, beta, out + offset * n,
       ldc); offset += m / granularity;
            }
        }
    */
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

        auto ex = oneapi::mkl::blas::row_major::gemm(
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

int main() {

    ScheduleEngine SE(4);
    SE.print_engine_info();
    int m, k, n;
    printf("Enter m,n,k\n");
    scanf("%d %d %d", &m, &k, &n);
    printf("Creating Input\n");
    Buffer<float> input(m * k, &SE);
    input.print_host_data();
    printf("Creating Weights\n");
    Buffer<float> weights(k * n, &SE);
    printf("Creating Output\n");
    Buffer<float> output(m * n, &SE);
    printf("Creating FeedForward Instance\n");
    FeedForward<float> qkv_linear(FeedForward<float>::Config(m, n, k));
    std::cout << "Input: \n";
    input.print_host_data();
    std::cout << "Weights: \n";
    weights.print_host_data();

    //    qkv_linear.Forward(&input, &weights, &output, &SE);
    return 0;
}
