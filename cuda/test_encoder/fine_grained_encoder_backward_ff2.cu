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
    int print_info=0; 

    int bsz = batch_size * sequence_length;

    bool _gelu_checkpoint = false;
    bool _pre_or_postLayerNorm = false;

    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};

    if(print_info){
        std::cout << "################################################################" << std::endl;
        std::cout << "batch size=" << batch_size << std::endl;
        std::cout << "sequence length=" << sequence_length << std::endl;
        std::cout << "hidden layer size=" << hidden_size << std::endl;
        std::cout << "intermediate size=" << intermediate_size << std::endl;
        std::cout << "number of queues=" << nq << std::endl;
        std::cout << "################################################################" << std::endl;
    }

    Stopwatch sw, sw1;
    ScheduleEngine SE(8);

    Dropout <float> _layer_output_dropout(Dropout<float>::Config(0.2, hidden_size));

    Buffer<float> buf_0(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> buf_1(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> buf_2(batch_size * sequence_length * hidden_size + bsz * sequence_length * hidden_size, &SE);
    Buffer<float> ff2_inp_ptr(batch_size * sequence_length * intermediate_size, &SE);
    Buffer<float> output_w_ptr(intermediate_size * hidden_size, &SE);
    Buffer<float> grad_output_w_ptr(intermediate_size * hidden_size, &SE);
    Buffer<float> grad_output_b_ptr(hidden_size, &SE);
    Buffer<float> ff2_buf(batch_size * sequence_length * hidden_size + 3 * bsz * sequence_length * hidden_size, &SE);
    Buffer<float> grad_output_ptr(batch_size * sequence_length * hidden_size, &SE);

    Buffer<float> layer_dropout_buf = _layer_output_dropout.HasDropout()
                                      ? buf_0
                                      : (_pre_or_postLayerNorm ? grad_output_ptr : buf_1); 

    FeedForward<float> _ff2(FeedForward<float>::Config(bsz, 
                                                       hidden_size,
                                                       intermediate_size,
                                                       gemm_algos,
                                                       true));

    sw.start();
    _ff2.BackwardFineGrained(bsz,
                    nq,
                    &layer_dropout_buf,
                    (_gelu_checkpoint ? &buf_2 : &ff2_inp_ptr),
                    &output_w_ptr,
                    &grad_output_w_ptr,
                    &grad_output_b_ptr,
                    &SE,
                    &ff2_buf);
    sw.stop();
    printf("_ff2.Backward(): %f\n", sw.GetTimeInSeconds());

    return 0;
}