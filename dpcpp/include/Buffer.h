#ifndef BUFFER_H
#define BUFFER_H
#include <vector>
#include <CL/sycl.hpp>
#include <stdio.h>
using namespace cl::sycl;

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

template <typename T> 
class Buffer {
  public:
    Buffer(int num_elements, ) {
      
      auto device = streams[0].get_device();
      auto context = streams[0].get_context();

      T *_host_data=(float *)malloc(sizeof(T)*num_elements);
      T *_device_data=(float *)malloc_device(sizeof(T)* num_elements,device,context);
        

    }
     
    ~Buffer() {

    }


  private:
    T * _host_data;
    T *_device_data;
  
#if FILE_LOGGER
    FILE *fp;
#endif
};
#endif
