#include "gelu.h"
#include "gemm.h"
#include "utils.h"
#include "dropout.h"
#include "softmax.h"
#include "normalize.h"
#include "transform.h"
#include "strided_gemm.h"

int main(int argc, char* argv[]) {
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int intermediate_size = atoi(argv[4]);
    int nq = atoi(argv[6]);

    float layernorm_eps=0.000001;     

    std::cout << "################################################################" << std::endl;
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "intermediate size=" << intermediate_size << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
    std::cout << "################################################################" << std::endl;

    Stopwatch sw;
    ScheduleEngine SE(8);

    int bsz = batch_size * sequence_length;

    Buffer<float> norm_var(batch_size * sequence_length, &SE);
    Buffer<float> norm_mean(batch_size * sequence_length, &SE);
    Buffer<float> attn_norm_var(batch_size * sequence_length, &SE);
    Buffer<float> attn_norm_mean(batch_size * sequence_length, &SE);
    Buffer<float> grad_output_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> norm_w_ptr(hidden_size, &SE);
    Buffer<float> grad_norm_w_ptr(hidden_size, &SE);
    Buffer<float> grad_norm_b_ptr(hidden_size, &SE);
    Buffer<float> buf_1(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> inp_norm_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> norm_b_ptr(hidden_size, &SE);
    Buffer<float> output_ptr(batch_size * sequence_length * hidden_size, &SE);

    Normalize<float> _layer_norm(Normalize<float>::Config(batch_size, 
                                                               sequence_length,
                                                               hidden_size,
                                                               layernorm_eps,
                                                               true));

    _layer_norm.SetMeansAndVariance(&norm_mean, &norm_var);

    sw.start();
    _layer_norm.BackwardFineGrained(bsz,
                                    nq,
                                    &grad_output_ptr,
                                    &norm_w_ptr,
                                    &grad_norm_w_ptr,
                                    &grad_norm_b_ptr,
                                    &SE,
                                    &buf_1,
                                    &inp_norm_ptr,
                                    true);
    sw.stop();
    printf("_layer_norm.BackwardFineGrained(): %f\n", sw.GetTimeInSeconds());
                                           
    return 0;
}