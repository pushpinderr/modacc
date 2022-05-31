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
    int nh = atoi(argv[5]);
    int nq = atoi(argv[6]);

    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
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

    Dropout <float> _attn_output_dropout(Dropout<float>::Config(0.2, hidden_size));
    Buffer<float> buf_0(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> buf_1(batch_size * sequence_length * hidden_size, &SE);    
    Buffer<float> buf_2(batch_size * sequence_length * hidden_size + bsz * sequence_length * hidden_size, &SE);

    Buffer<float> attn_output_dropout_buf = _attn_output_dropout.HasDropout() ? buf_2 : buf_0;
    Buffer<float> attn_o_inp_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_ow_ptr(batch_size * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_ow_ptr(batch_size * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_ob_ptr(hidden_size, &SE);

    FeedForward<float> _attn_out_linear(FeedForward<float>::Config(bsz, 
                                                              hidden_size, 
                                                              hidden_size,
                                                              gemm_algos,
                                                              true));
    sw.start();   
    _attn_out_linear.BackwardFineGrained(bsz,
                            nq,
                            &attn_output_dropout_buf,
                            &attn_o_inp_ptr,
                            &attn_ow_ptr,
                            &grad_attn_ow_ptr,
                            &grad_attn_ob_ptr,
                            &SE,
                            &buf_1,
                            true);
    sw.stop();
    printf("_attn_out_linear.BackwardFineGrained(): %f\n", sw.GetTimeInSeconds());

    return 0;
}