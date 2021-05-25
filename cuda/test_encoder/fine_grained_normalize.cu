#include "utils.h"
#include "normalize.h"

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
    float layernorm_eps=0.000001; 

    ScheduleEngine SE(30);
    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> input_norm(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> norm_weights(hidden_size, &SE);
    Buffer<float> norm_bias(hidden_size, &SE);
    Buffer<float> norm_var(batch_size * sequence_length, &SE);
    Buffer<float> norm_mean(batch_size * sequence_length, &SE);

    
    Normalize<float> _normalize(Normalize<float>::Config(batch_size, sequence_length, hidden_size, layernorm_eps, true));
    _normalize.SetMeansAndVariance(&norm_mean, &norm_var);

    Stopwatch sw;

    Buffer<float> output(3 * hidden_size * batch_size * sequence_length, &SE); 
    sw.start();
    _normalize.ForwardCheckpointPartition(batch_size*sequence_length, nq, &input_norm, &input, &norm_weights, &norm_bias, &SE, sync_flag);
    sw.stop();
    std::cout << "t=" << sw.GetTimeInSeconds() << std::endl; 
    fileWrite("queue_size="+std::to_string(nq)+".txt", std::to_string(sw.GetTimeInSeconds()));

    return 0;
}
