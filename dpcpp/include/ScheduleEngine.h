#ifndef SCHEDULEENGINE_H
#define SCHEDULEENGINE_H
#include <CL/sycl.hpp>
#include <stdio.h>
using namespace cl::sycl;

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"

class ScheduleEngine {
  public:
    ScheduleEngine(int num_queues) {
        this->num_queues = num_queues;
        _devices = device::get_devices(info::device_type::gpu);

        _devices.erase(
            std::remove_if(_devices.begin(), _devices.end(),
                           [](auto Device) { return Device.is_host(); }),
            __devices.end());

        _context = context(_devices, &exceptionHandler);
        for (int i = 0; i < num_queues; i++)
            _queues.push_back(queue(context, _devices[0],property::queue::in_order()));
    }

    ~ScheduleEngine() {}

  private:
    std::vector<device> _devices;
    context _context;
    int num_queues;
    std::vector<queue> _queues;
};
#endif
