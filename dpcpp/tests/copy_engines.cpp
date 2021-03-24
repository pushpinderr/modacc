#include <CL/sycl.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
using namespace cl::sycl;

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

bool verifyProfiling(event Event) {
    auto Submit =
        Event.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto Start =
        Event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto End =
        Event.get_profiling_info<sycl::info::event_profiling::command_end>();

    assert(Submit <= Start);
    assert(Start <= End);
//    std::cout << Start << " " << End << "\n";
    bool Pass = sycl::info::event_command_status::complete ==
                Event.get_info<sycl::info::event::command_execution_status>();

    return Pass;
}

int main(int argc, char *argv[]) {
    int num_queues = atoi(argv[1]);
    int hidden_size = 1024;
    int qkv_size = 3072;
    int num_elements = hidden_size * qkv_size;
    
    std::vector<unsigned long> start_times, end_times;

    std::vector<queue> copy_engines;
    auto _devices = device::get_devices(info::device_type::gpu);

    _devices.erase(std::remove_if(_devices.begin(), _devices.end(),
                                  [](auto Device) { return Device.is_host(); }),
                   _devices.end());

    auto _context = context(_devices[0], &exceptionHandler);
    for (int i = 0; i < num_queues; i++)
        copy_engines.push_back(
            queue(_context, _devices[0], property::queue::enable_profiling()));

    float *_host_data = (float *)malloc(sizeof(float) * num_elements);
    float *_device_data = (float *)malloc_device(sizeof(float) * num_elements,
                                                 _devices[0], _context);

    event ev[num_queues];
    int num_sub_elements = num_elements / num_queues;
    int subSize = sizeof(float) * num_elements / num_queues;
    for (int i = 0; i < num_queues; i++)
        ev[i] =
            copy_engines[i].memcpy(_device_data + i, _host_data + i, subSize);

    for (int i = 0; i < num_queues; i++)
        copy_engines[i].wait();
    for (int i = 0; i < num_queues; i++)
    {
        verifyProfiling(ev[i]);
        start_times.push_back(ev[i].get_profiling_info<sycl::info::event_profiling::command_start>());
        end_times.push_back(ev[i].get_profiling_info<sycl::info::event_profiling::command_end>());
        

    }

    for (int i = 0; i < num_queues; i++)
    {
        std::cout<<start_times[i]-start_times[0] <<" "<<end_times[i]-start_times[0]<<"\n";
    }
    printf("======================================\n");
}
