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

    int bsz = batch_size * sequence_length;

    int print_info=0; 

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

    Stopwatch sw;
    ScheduleEngine SE(8);

    Buffer<float> buf_2(batch_size * sequence_length * hidden_size + bsz * sequence_length * hidden_size, &SE);
    Buffer<float> ff2_buf(batch_size * sequence_length * hidden_size + 3 * bsz * sequence_length * hidden_size, &SE);
    Buffer<float> inp_norm_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_qkvw_ptr(3 * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_qkvw_ptr(3 * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_qkvb_ptr(3 * hidden_size, &SE);
    Buffer<float> input_ptr(batch_size * sequence_length * hidden_size, &SE);

    FeedForward<float> _qkv_linear(FeedForward<float>::Config(bsz, 
                                                              3 * hidden_size, 
                                                              hidden_size,
                                                              gemm_algos,
                                                              true));

    sw.start();
    if (_pre_or_postLayerNorm) {
       _qkv_linear.BackwardFineGrained(bsz,
                            nq,
                            &ff2_buf,
                            &inp_norm_ptr,
                            &attn_qkvw_ptr,
                            &grad_attn_qkvw_ptr,
                            &grad_attn_qkvb_ptr,
                            &SE,
                            &buf_2,
                            true); 
    } else {
        _qkv_linear.BackwardFineGrained(bsz,
                            nq,
                            &ff2_buf,
                            &input_ptr,
                            &attn_qkvw_ptr,
                            &grad_attn_qkvw_ptr,
                            &grad_attn_qkvb_ptr,
                            &SE,
                            &buf_2,
                            true);
    }
    sw.stop();
    printf("_qkv_linear.Backward(): %f\n", sw.GetTimeInSeconds());

    return 0;
}