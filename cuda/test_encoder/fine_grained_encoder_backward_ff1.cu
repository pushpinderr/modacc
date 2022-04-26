#include "gelu.h"
#include "gemm.h"
#include "utils.h"
#include "dropout.h"
#include "softmax.h"
#include "normalize.h"
#include "transform.h"
#include "strided_gemm.h"

int main(int argc, char* argv[]) {

    int nq = atoi(argv[6]);
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int intermediate_size = atoi(argv[4]);

    int bsz_seq = batch_size * sequence_length;

    Stopwatch sw;
    ScheduleEngine SE(8);

    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
    
    std::cout << "################################################################" << std::endl;
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "intermediate size=" << intermediate_size << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
    std::cout << "################################################################" << std::endl;
    
    FeedForward<float> _ff1(FeedForward<float>::Config(bsz_seq, 
                                                       intermediate_size,
                                                       hidden_size,
                                                       gemm_algos,
                                                       true));

    Buffer<float> ff2_buf(batch_size * sequence_length * hidden_size + 3 * bsz_seq * sequence_length * hidden_size, &SE);
    Buffer<float> ff1_inp_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> inter_w_ptr(hidden_size * intermediate_size, &SE);
    Buffer<float> grad_inter_w_ptr(hidden_size * intermediate_size, &SE);
    Buffer<float> grad_inter_b_ptr(intermediate_size, &SE);
    Buffer<float> buf_3(batch_size * sequence_length * hidden_size + 2 * bsz_seq * sequence_length * hidden_size, &SE);

    sw.start();
    _ff1.BackwardFineGrained(bsz_seq,
                            nq,
                            &ff2_buf,
                            &ff1_inp_ptr,
                            &inter_w_ptr,
                            &grad_inter_w_ptr,
                            &grad_inter_b_ptr,
                            &SE,
                            &buf_3,
                            true);
    sw.stop();
    printf("_ff1.BackwardFineGrained(): %f\n", sw.GetTimeInSeconds());

    return 0;
}