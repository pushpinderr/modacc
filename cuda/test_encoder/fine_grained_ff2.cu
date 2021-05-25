// %%cuda --name fine_grained.cu
#include "gemm.h"
#include "utils.h"


int main(int argc, char *argv[])
{
    // PLEASE NOTE: number of queues must be less than batch_size
    bool sync_flag = true;
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int intermediate_size = atoi(argv[4]);
    int nh = atoi(argv[5]);
    int nq = atoi(argv[6]);
    int bsz=batch_size*sequence_length;
    int bsz_seq=batch_size*sequence_length;
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
    Buffer<float> context(batch_size * sequence_length * sequence_length * nh, &SE); // not sure about this.
    Buffer<float> output_w(intermediate_size * hidden_size, &SE);
    Buffer<float> out(batch_size * sequence_length * hidden_size, &SE);
    
    FeedForward<float> _ff2(FeedForward<float>::Config(bsz, 
                                                       hidden_size,
                                                       intermediate_size,
                                                       gemm_algos,
                                                       true));
     Stopwatch sw;
    
    printf("\x1b[41;1mstarting profiling for fine grained implementation\x1b[0m\n");
    sw.restart();
    _ff2.ForwardCheckpointPartition(bsz_seq, &context, &output_w, &out, &SE, nq, true);
    sw.stop();
    std::cout << "t(" << nq << ")=" << sw.GetTimeInSeconds() << std::endl;
    fileWrite("queue_size="+std::to_string(nq)+".txt", std::to_string(sw.GetTimeInSeconds()));

    return 0;
}
