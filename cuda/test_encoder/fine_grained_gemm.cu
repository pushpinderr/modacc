// %%cuda --name fine_grained.cu
#include "gemm.h"
#include "utils.h"


int main(int argc, char *argv[])
{
    // PLEASE NOTE: number of queues must be less than batch_size
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int nq = atoi(argv[4]);
    bool sync_flag = true;

    /* if ( argc > 5 )
    {
        if ( std::string(argv[5]) == "sync" )
            sync_flag = true;
    } */

    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
    std::cout << "################################################################" << std::endl;
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "sync flag=" << sync_flag << std::endl;
    std::cout << "################################################################" << std::endl;

    ScheduleEngine SE(30);
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> weights(3 * hidden_size * hidden_size, &SE);
    FeedForward<float> _qkv_linear(FeedForward<float>::Config(batch_size * sequence_length, 3 * hidden_size, hidden_size, gemm_algos, true));
    
    Stopwatch sw;
    
    printf("\x1b[41;1mstarting profiling for fine grained implementation\x1b[0m\n");
    Buffer<float> output(3 * hidden_size * batch_size * sequence_length, &SE);
    sw.restart();
    _qkv_linear.ForwardCheckpointPartition(batch_size * sequence_length, &input, &weights, &output, &SE, nq, sync_flag);
    sw.stop();
    std::cout << "t(" << nq << ")=" << sw.GetTimeInSeconds() << std::endl;
    fileWrite("queue_size="+std::to_string(nq)+".txt", std::to_string(sw.GetTimeInSeconds()));

    return 0;
}