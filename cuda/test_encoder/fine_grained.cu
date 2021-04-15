%%cuda --name fine_grained.cu
#include "gemm.h"
#include "utils.h"


int main(int argc, char *argv[])
{
    // PLEASE NOTE: number of queues must be less than batch_size
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
 
    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
 
    ScheduleEngine SE(16);
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> weights(3 * hidden_size * hidden_size, &SE);
    Buffer<float> output(3 * hidden_size * batch_size * sequence_length, &SE); 
    FeedForward<float> _qkv_linear(FeedForward<float>::Config(batch_size * sequence_length, 3 * hidden_size, hidden_size, gemm_algos, true));
    
    Stopwatch sw;

    printf("\x1b[41;1mstarting profiling for coarse grained implementation\x1b[0m\n");
    output.init(1);
    
    sw.start();
    _qkv_linear.ForwardCheckpoint(batch_size * sequence_length, &input, &weights, &output, &SE, false);
    sw.stop();
    std::cout << "time taken=" << sw.GetTimeInSeconds() << std::endl;

    printf("\x1b[41;1mstarting profiling for fine grained implementation\x1b[0m\n");
    for ( int i = 2; i <= 16; i*=2 )
    {   
        std::cout << "number of queues=" << i << std::endl;
        output.init(1);
        sw.restart();
        _qkv_linear.ForwardCheckpointPartition(batch_size * sequence_length, &input, &weights, &output, &SE, i, true);
        sw.stop();
        std::cout << "time taken=" << sw.GetTimeInSeconds() << std::endl;
    }

    return 0;
}