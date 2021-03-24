GEMM Experiments 
--------------
* The code test_engine.cpp located inside dpcpp/tests runs the oneMKL GEMM kernel on both CPU and GPU devices multiple number of times and profiles execution using the events API in DPC++. The objective of this test is to evaluate the performance of i) fine grained scheduling i.e. partitioning the computation of GEMM by using multiple queues for the same device and ii) partitioning GEMM computationg acorss a single CPU and a single GPU device. The corresponding functions involved are

  1. ForwardPartitionWeights: Fine grained scheduling
  2. ForwardPartitionWeights_CPU_GPU: Partitioned CPU/GPU execution
* Change directory to modacc/dpcpp.
* For compiling run bash build.sh. This would run the script build_test.sh located inside the scripts folder. The compute node to which this script would be launched is specified in the build.sh file located inside dpcpp folder.  
* For executing run bash run.sh. This would run the script run_test.sh located inside the scripts folder. The run_test.sh file contains a list of commands pertaining to the following format.
```
./tests/./test_engine batch_size sequence_length hidden_size 3 $\times$ hidden_size number_of_queues mode
```
Here, mode can take values 0,1,2 where 0 engages the kernel to the CPU device, 1 engages the kernel to the GPU device and 2 engages the kernel to both the CPU and GPU devices. For mode=0,1, number_of_queues dictate the total number of queues across which the matrix computation can be partitioned. For mode=2, number_of_queues actually dictates how the computation can be partitioned across the CPU and GPU device. For example if number_of_queues=4, 1/4th of the computation is processed onthe GPU device while 3/4th is processed on the CPU device. For the case of matrix multiplication, this translates to partitioning the number of rows of one matrix in 1:4 ratio across the CPU and GPU device. 
