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
    bool sync = true;
    float layernorm_eps = 0.000001; 
    bool _pre_or_postLayerNorm = false;
    bool _gelu_checkpoint = true;
    bool _attn_dropout_checkpoint= false;

    cublasHandle_t _cublasHandle;

    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
    std::cout << "+----------------------------------------------------------+" << std::endl;
    std::cout << "|  Batch Size         :   "<< batch_size <<"                                |"<< std::endl;
    std::cout << "|  Sequence Length    :   "<< sequence_length<<"                              |"<< std::endl;
    std::cout << "|  Hidden Layer Size  :   "<< hidden_size <<"                              |"<< std::endl;
    std::cout << "|  Intermediate Size  :   "<< intermediate_size <<"                              |"<< std::endl;
    std::cout << "|  Number of Heads    :   "<< nh <<"                                |"<< std::endl;
    std::cout << "|  Number of Queues   :   "<< nq <<"                                |"<<std::endl;
    std::cout << "|  Sync Flag          :   "<< sync <<"                                |"<<std::endl;
    std::cout << "+----------------------------------------------------------+" << std::endl;

    Stopwatch sw, sw1;
    ScheduleEngine SE(8);

    int bsz = batch_size * sequence_length;
    int bsz_seq = batch_size * sequence_length;
    int bsz_heads = batch_size * nh;
    size_t small_buf_size = bsz * sequence_length * hidden_size;

    FeedForward<float> _qkv_linear(FeedForward<float>::Config(bsz, 
                                                              3 * hidden_size, 
                                                              hidden_size,
                                                              gemm_algos,
                                                              true));

    FeedForward<float> _attn_out_linear(FeedForward<float>::Config(bsz, 
                                                              hidden_size, 
                                                              hidden_size,
                                                              gemm_algos,
                                                              true));

    Normalize<float> _attn_layer_norm(Normalize<float>::Config(batch_size, 
                                                               sequence_length,
                                                               hidden_size,
                                                               layernorm_eps,
                                                               true));

    Normalize<float> _layer_norm(Normalize<float>::Config(batch_size, 
                                                               sequence_length,
                                                               hidden_size,
                                                               layernorm_eps,
                                                               true));

    FeedForward<float> _ff1(FeedForward<float>::Config(bsz, 
                                                       intermediate_size,
                                                       hidden_size,
                                                       gemm_algos,
                                                       true));

    FeedForward<float> _ff2(FeedForward<float>::Config(bsz, 
                                                       hidden_size,
                                                       intermediate_size,
                                                       gemm_algos,
                                                       true));

    Softmax<float> _softmax(Softmax<float>::Config(batch_size, nh, sequence_length));
    
    Gelu<float> _gelu{Gelu<float>::Config(intermediate_size)};
    
    Dropout<float> _attn_prob_dropout(Dropout<float>::Config(0.2, sequence_length));
    
    Dropout <float> _attn_output_dropout(Dropout<float>::Config(0.2, hidden_size));
    
    Dropout <float> _layer_output_dropout(Dropout<float>::Config(0.2, hidden_size));

    StridedBatchGemm<float> _attn_scores(StridedBatchGemm<float>::Config(batch_size * nh,
                                                                         sequence_length,
                                                                         sequence_length,
                                                                         hidden_size / nh,
                                                                         1.0 / (sqrt(hidden_size / nh)),
                                                                         0.0,
                                                                         CUBLAS_OP_T,
                                                                         CUBLAS_OP_N,
                                                                         gemm_algos));

    StridedBatchGemm<float> _attn_context(StridedBatchGemm<float>::Config(batch_size * nh,
                                                         hidden_size / nh,
                                                         sequence_length,
                                                         sequence_length,
                                                         1.0,
                                                         0.0,
                                                         CUBLAS_OP_N,
                                                         CUBLAS_OP_N,
                                                         gemm_algos));

    Buffer<float> norm_var(batch_size * sequence_length, &SE);
    Buffer<float> norm_mean(batch_size * sequence_length, &SE);
    Buffer<float> attn_norm_var(batch_size * sequence_length, &SE);
    Buffer<float> attn_norm_mean(batch_size * sequence_length, &SE);
    Buffer<float> inter_b(intermediate_size, &SE);
    
    Buffer<float> grad_output_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> grad_norm_w_ptr(hidden_size, &SE);
    Buffer<float> grad_norm_b_ptr(hidden_size, &SE);
    Buffer<float> output_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> ff1_inp_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> inter_w_ptr(hidden_size * intermediate_size, &SE);
    Buffer<float> grad_inter_w_ptr(hidden_size * intermediate_size, &SE);
    Buffer<float> grad_inter_b_ptr(intermediate_size, &SE);

    Buffer<float> attn_nw_ptr(hidden_size, &SE);
    Buffer<float> attn_nb_ptr(hidden_size, &SE);
    Buffer<float> grad_attn_nw_ptr(hidden_size, &SE);
    Buffer<float> grad_attn_nb_ptr(hidden_size, &SE);
    Buffer<float> add_res_ptr(batch_size * sequence_length * hidden_size, &SE);

    Buffer<float> attn_o_inp_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_ow_ptr(batch_size * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_ow_ptr(batch_size * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_ob_ptr(hidden_size, &SE);

    Buffer<float> soft_out_ptr(batch_size * sequence_length * hidden_size + 4 * bsz * sequence_length * hidden_size, &SE);
    //Buffer<float> soft_out_ptr(batch_size * sequence_length * sequence_length * nh, &SE);
   
    Buffer<float> ctx_bufB_ptr(batch_size * sequence_length * sequence_length * nh, &SE);
    Buffer<float> q_tf_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> k_tf_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> v_tf_ptr(batch_size * sequence_length * hidden_size, &SE);

    Buffer<float> inp_norm_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> grad_attn_qkvb_ptr(3 * hidden_size, &SE);
    Buffer<float> input_ptr(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_qkvw_ptr(3 * hidden_size * hidden_size, &SE);
    Buffer<float> grad_attn_qkvw_ptr(3 * hidden_size * hidden_size, &SE);
    Buffer<float> norm_w_ptr(hidden_size, &SE);

    Buffer<float> grad_input_ptr(batch_size * sequence_length * hidden_size + 2 * bsz * sequence_length * hidden_size, &SE);

//    Buffer<float> grad_input_ptr(batch_size * sequence_length * hidden_size, &SE);
    
    Buffer<float> norm_b_ptr(hidden_size, &SE);

    Buffer<float> ff2_inp_ptr(batch_size * sequence_length * intermediate_size, &SE);
    Buffer<float> gelu_inp_ptr(batch_size * sequence_length * intermediate_size, &SE);
    
    Buffer<float> output_w_ptr(intermediate_size * hidden_size, &SE);
    Buffer<float> grad_output_w_ptr(intermediate_size * hidden_size, &SE);
    Buffer<float> grad_output_b_ptr(hidden_size, &SE);

    Buffer<float> buf_0(batch_size * sequence_length * hidden_size + 2 * bsz * sequence_length * hidden_size, &SE);

    //Buffer<float> buf_0(batch_size * sequence_length * hidden_size, &SE);
    // Buffer<float> buf_1(batch_size * sequence_length * hidden_size, &SE);

    Buffer<float> buf_1(batch_size * sequence_length * hidden_size + 4 * bsz * sequence_length * hidden_size, &SE);

   //Buffer<float> buf_1(batch_size * sequence_length * hidden_size + bsz * sequence_length * hidden_size, &SE);

 //   Buffer<float> buf_2(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> buf_2(batch_size * sequence_length * hidden_size + 2 * bsz * sequence_length * hidden_size, &SE);
    Buffer<float> buf_3(batch_size * sequence_length * hidden_size + 3 * bsz * sequence_length * hidden_size, &SE);

    // Buffer<float> ff2_buf(batch_size * sequence_length * hidden_size, &SE);

    Buffer<float> ff2_buf(batch_size * sequence_length * hidden_size + 4 * bsz * sequence_length * hidden_size, &SE);
    Buffer<float> ctx_bufB_ptr_recomp(batch_size * sequence_length * hidden_size + 3 * bsz * sequence_length * hidden_size + (sequence_length * sequence_length * bsz * bsz_heads), &SE);

    Buffer<uint8_t> attn_output_dropout_mask(batch_size * sequence_length * hidden_size, &SE); 

    _layer_norm.SetMeansAndVariance(&norm_mean, &norm_var);
    _attn_layer_norm.SetMeansAndVariance(&attn_norm_mean, &attn_norm_var);
    //_attn_output_dropout.SetMask(&attn_output_dropout_mask);

    sw1.start();
    printf("Fine Grained Backward propogation starts: %f\n", sw1.GetTimeInSeconds());

    if (!_pre_or_postLayerNorm) {
        sw.start();
        if (_layer_norm.UseMean()) {
            _layer_norm.BackwardFineGrained(bsz_seq,
                                    nq,
                                    &grad_output_ptr,
                                    &norm_w_ptr,
                                    &grad_norm_w_ptr,
                                    &grad_norm_b_ptr,
                                    &SE,
                                    &buf_1,
                                    &inp_norm_ptr,
                                    true);
                                    
        } else {
            _layer_norm.BackwardFineGrained(bsz_seq,
                                    nq,
                                    &grad_output_ptr,
                                    &norm_w_ptr,
                                    &norm_b_ptr,
                                    &grad_norm_w_ptr,
                                    &grad_norm_b_ptr,
                                    &SE,
                                    &buf_1,
                                    &output_ptr,
                                    true);
        }
        CHECK(cudaDeviceSynchronize());        
        sw.stop();
        printf("_layer_norm.BackwardFineGrained(): %f\n\n", sw.GetTimeInSeconds());
    }


    sw.restart();
    if (_pre_or_postLayerNorm) {
        _layer_output_dropout.Backward(bsz_seq, &buf_0, &grad_output_ptr,  &attn_output_dropout_mask, &SE);
    } else {
        _layer_output_dropout.Backward(bsz_seq, &buf_0, &buf_1, &attn_output_dropout_mask, &SE);
    }       
    CHECK(cudaDeviceSynchronize());
    sw.stop();
    printf("_layer_output_dropout.Backward(): %f\n", sw.GetTimeInSeconds());

    Buffer<float> layer_dropout_buf = _layer_output_dropout.HasDropout()
                                      ? buf_0
                                      : (_pre_or_postLayerNorm ? grad_output_ptr : buf_1);

    if (_gelu_checkpoint) {
        sw.restart();
        _gelu.ForwardWithBiasAdd(bsz_seq, &ff2_inp_ptr, &inter_b, &buf_2, &SE);
        CHECK(cudaDeviceSynchronize());
        sw.stop();
        printf("_gelu.ForwardWithBiasAdd(): %f\n", sw.GetTimeInSeconds());
    }

    sw.restart();
    _ff2.BackwardFineGrained(bsz_seq,
                    nq,
                    &layer_dropout_buf,
                    &ff2_inp_ptr,
                    &output_w_ptr,
                    &grad_output_w_ptr,
                    &grad_output_b_ptr,
                    &SE,
                    &ff2_buf);
    CHECK(cudaDeviceSynchronize());
    sw.stop();
    printf("_ff2.BackwardFineGrained(): %f\n", sw.GetTimeInSeconds());

    sw.restart();
    _gelu.Backward(bsz_seq, 
                            &ff2_buf,
                            &gelu_inp_ptr,
                            &inter_b,
                            &SE);
    CHECK(cudaDeviceSynchronize());
    sw.stop();
    printf("_gelu.Backward(): %f\n", sw.GetTimeInSeconds());

    sw.restart();
    _ff1.BackwardFineGrained(bsz_seq,
                            nq,
                            &ff2_buf,
                            &ff1_inp_ptr,             
                            &inter_w_ptr,
                            &grad_inter_w_ptr,
                            &grad_inter_b_ptr,
                            &SE,
                            &buf_3);
    CHECK(cudaDeviceSynchronize());                            
    sw.stop();
    printf("_ff1.BackwardFineGrained(): %f\n", sw.GetTimeInSeconds());

    // if (!_pre_or_postLayerNorm) {
    //     sw.restart();
    //     launch_fused_add2<float>(&buf_2, &buf_3, &buf_1, batch_size, sequence_length, hidden_size, &SE);
    //     CHECK(cudaDeviceSynchronize());                            
    //     sw.stop();
    //     printf("launch_fused_add2(): %f\n", sw.GetTimeInSeconds());        
    // }

    // sw.restart();
    // if (_pre_or_postLayerNorm) {
    //     if (_attn_layer_norm.UseMean()) {
    //         _attn_layer_norm.BackwardFusedAdd(bsz_seq,
    //                                         &buf_3,
    //                                         &grad_output_ptr,
    //                                         &attn_nw_ptr,
    //                                         &grad_attn_nw_ptr,
    //                                         &grad_attn_nb_ptr,
    //                                         &SE,
    //                                         &buf_0,
    //                                         &add_res_ptr);
    //     } else {
    //         _attn_layer_norm.BackwardFusedAdd(bsz_seq,
    //                                         &buf_3,
    //                                         &grad_output_ptr,
    //                                         &attn_nw_ptr,
    //                                         &attn_nb_ptr,
    //                                         &grad_attn_nw_ptr,
    //                                         &grad_attn_nb_ptr,
    //                                         &SE,
    //                                         &buf_0,
    //                                         &ff1_inp_ptr);
    //     }
    // } else {
    //     if (_attn_layer_norm.UseMean()) {
    //         _attn_layer_norm.BackwardFineGrained(bsz_seq,
    //                                             nq,
    //                                             &buf_2,
    //                                             &attn_nw_ptr,
    //                                             &grad_attn_nw_ptr,
    //                                             &grad_attn_nb_ptr,
    //                                             &SE,
    //                                             &buf_0,
    //                                             &add_res_ptr);
    //     } else {
    //         _attn_layer_norm.BackwardFineGrained(bsz_seq,
    //                                             nq,
    //                                             &buf_2,
    //                                             &attn_nw_ptr,
    //                                             &attn_nb_ptr,
    //                                             &grad_attn_nw_ptr,
    //                                             &grad_attn_nb_ptr,
    //                                             &SE,
    //                                             &buf_0,
    //                                             &ff1_inp_ptr);
    //     }       
    // }
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_attn_layer_norm.Backward(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // _attn_output_dropout.BackwardFineGrained(bsz_seq, nq, &buf_2, &buf_0,  &attn_output_dropout_mask, &SE);
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_attn_output_dropout.Backward(): %f\n", sw.GetTimeInSeconds());
    
    // Buffer<float> attn_output_dropout_buf = _attn_output_dropout.HasDropout() ? buf_2 : buf_0;

    // sw.restart();   
    // _attn_out_linear.BackwardFineGrained(bsz_seq,
    //                         nq,
    //                         &attn_output_dropout_buf,
    //                         &attn_o_inp_ptr,
    //                         &attn_ow_ptr,
    //                         &grad_attn_ow_ptr,
    //                         &grad_attn_ob_ptr,
    //                         &SE,
    //                         &buf_1);
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_attn_out_linear.Backward(): %f\n", sw.GetTimeInSeconds());

    // sw.restart(); 
    // launch_transform_0213<float>(&buf_2, &buf_1, batch_size, sequence_length, hidden_size, nh, &SE);  
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("launch_transform_0213(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // if (_attn_prob_dropout.HasDropout()) {
    //     if (_attn_dropout_checkpoint) {
    //         _attn_prob_dropout.Forward(bsz_heads * sequence_length, 
    //                                     &ctx_bufB_ptr_recomp, 
    //                                     &soft_out_ptr, 
    //                                     &SE, 
    //                                     true);
    //     }
        
    //     _attn_context.Backward(bsz_heads,
    //                         &buf_2,
    //                         &v_tf_ptr,
    //                         (_attn_dropout_checkpoint ? &ctx_bufB_ptr_recomp : &ctx_bufB_ptr),
    //                         &SE,
    //                         &buf_3,
    //                         &ff2_buf);
    // } else {
    //     _attn_context.Backward(bsz_heads, 
    //                             &buf_2, 
    //                             &v_tf_ptr, 
    //                             &soft_out_ptr, 
    //                             &SE, 
    //                             &buf_3, 
    //                             &ff2_buf);
    // }
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_attn_context.Backward(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // // _attn_prob_dropout.BackwardFineGrained(bsz_heads * sequence_length, 
    // //                                         nq, 
    // //                                         &ff2_buf,
    // //                                         &attn_output_dropout_mask, 
    // //                                         &SE);
    // _attn_prob_dropout.Backward(bsz_heads * sequence_length, 
    //                                         &ff2_buf,
    //                                         &attn_output_dropout_mask, 
    //                                         &SE);                                            
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_attn_prob_dropout.Backward(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // _softmax.Backward(bsz, 
    //                     &ff2_buf, 
    //                     &soft_out_ptr, 
    //                     &SE);
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_softmax.Backward(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // _attn_scores.Backward(bsz_heads, 
    //                         &ff2_buf, 
    //                         &k_tf_ptr, 
    //                         &q_tf_ptr, 
    //                         &SE, 
    //                         &buf_2, 
    //                         &buf_1);
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_attn_scores.Backward(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // launch_transform4d_0213(&ff2_buf, &buf_1, bsz, nh, sequence_length, hidden_size, &SE, 3);
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("launch_transform4d_0213(): %f\n", sw.GetTimeInSeconds());

    // sw.restart();
    // if (_pre_or_postLayerNorm) {
    //    _qkv_linear.BackwardFineGrained(bsz_seq,
    //                         nq,
    //                         &ff2_buf,
    //                         &inp_norm_ptr,
    //                         &attn_qkvw_ptr,
    //                         &grad_attn_qkvw_ptr,
    //                         &grad_attn_qkvb_ptr,
    //                         &SE,
    //                         &buf_2); 
    // } else {
    //     _qkv_linear.BackwardFineGrained(bsz_seq,
    //                         nq,
    //                         &ff2_buf,
    //                         &input_ptr,
    //                         &attn_qkvw_ptr,
    //                         &grad_attn_qkvw_ptr,
    //                         &grad_attn_qkvb_ptr,
    //                         &SE,
    //                         &buf_2);
    // }
    // CHECK(cudaDeviceSynchronize());
    // sw.stop();
    // printf("_qkv_linear.BackwardFineGrained(): %f\n", sw.GetTimeInSeconds());

    // if (_pre_or_postLayerNorm) {
    //     sw.restart();
    //     if (_layer_norm.UseMean()) {
    //         _layer_norm.BackwardFusedAdd(bsz_seq,
    //                                     &buf_2,
    //                                     &buf_0,
    //                                     &norm_w_ptr,
    //                                     &grad_norm_w_ptr,
    //                                     &grad_norm_b_ptr,
    //                                     &SE,
    //                                     &grad_input_ptr,
    //                                     &input_ptr);
    //     } else {
    //         _layer_norm.BackwardFusedAdd(bsz_seq,
    //                                     &buf_2,
    //                                     &buf_0,
    //                                     &norm_w_ptr,
    //                                     &norm_b_ptr,
    //                                     &grad_norm_w_ptr,
    //                                     &grad_norm_b_ptr,
    //                                     &SE,
    //                                     &grad_input_ptr,
    //                                     &inp_norm_ptr);
    //     }
    //     CHECK(cudaThreadSynchronize());
    //     sw.stop();
    //     printf("_layer_norm.BackwardFusedAdd(): %f\n", sw.GetTimeInSeconds());
    // } else {
    //     sw.restart();
    //     launch_fused_add2<float>(&grad_input_ptr, 
    //                                 &buf_2, 
    //                                 &buf_0, 
    //                                 bsz, 
    //                                 sequence_length, 
    //                                 hidden_size, 
    //                                 &SE);
    //     CHECK(cudaDeviceSynchronize());
    //     sw.stop();
    //     printf("launch_fused_add2(): %f\n", sw.GetTimeInSeconds());        
    // }

    sw1.stop();
    printf("Fine Grained Backward propogation ends: %f\n", sw1.GetTimeInSeconds());

    return 0;
}