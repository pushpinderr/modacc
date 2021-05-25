#include "gelu.h"
#include "gemm.h"
#include "utils.h"
#include "dropout.h"
#include "softmax.h"
#include "normalize.h"
#include "transform.h"
#include "strided_gemm.h"

/* template <typename T>
class BertEncoder {
public:
    struct Config {
        size_t batchSize;
        size_t heads;
        size_t seqLength;
        int num_heads;
        int intermediate_size;
        float hidden_output_dropout_ratio; 
        bool pre_or_postLayerNorm;
        std::vector<std::array<int, 3>> gemm_algos;
        bool attn_dropout_checkpoint; 
        bool normalize_invertible;
        bool gelu_checkpoint; 
        bool stochastic_mode;

        void read(std::string fname) {}
    }
    BertEncoder(Config config) : config_(config) {}
    ~BertEncoder() {}
    void Forward() {}

private:
    Config config_;
} */

int main(int argc, char* argv[]) {
    int batch_size = atoi(argv[1]);
    int sequence_length = atoi(argv[2]);
    int hidden_size = atoi(argv[3]);
    int intermediate_size = atoi(argv[4]);
    int nh = atoi(argv[5]);
    int nq = atoi(argv[6]);

    bool sync = true;
    float layernorm_eps=0.000001; 
    std::array <int, 3> gemm_algos = {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT};
    std::cout << "################################################################" << std::endl;
    std::cout << "batch size=" << batch_size << std::endl;
    std::cout << "sequence length=" << sequence_length << std::endl;
    std::cout << "hidden layer size=" << hidden_size << std::endl;
    std::cout << "intermediate size=" << intermediate_size << std::endl;
    std::cout << "number of heads=" << nh << std::endl;
    std::cout << "number of queues=" << nq << std::endl;
    std::cout << "sync flag=" << sync << std::endl;
    std::cout << "################################################################" << std::endl;

    Stopwatch sw;
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


    Buffer<float> input(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> input_norm(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> input_mask(batch_size * sequence_length * sequence_length * nh, &SE);
    Buffer<float> norm_weights(hidden_size, &SE);
    Buffer<float> norm_bias(hidden_size, &SE);
    Buffer<float> norm_var(batch_size * sequence_length, &SE);
    Buffer<float> norm_mean(batch_size * sequence_length, &SE);
    Buffer<float> qkv_weights(3 * hidden_size * hidden_size, &SE); 
    Buffer<float> qkv_bias(3 * hidden_size, &SE);
    Buffer<float> qkv_out(3 * hidden_size * batch_size * sequence_length, &SE);
    Buffer<float> keys(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> queries(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> values(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> soft_out(batch_size * sequence_length * sequence_length * nh, &SE);
    Buffer<float> context(batch_size * sequence_length * sequence_length * nh, &SE); // not sure about this.
    Buffer<uint8_t> attn_prob_dropout_mask(batch_size * sequence_length * sequence_length * nh, &SE); 
    Buffer<float> buf_1(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_o_inp(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_ow(batch_size * hidden_size * hidden_size, &SE);
    Buffer<float> add_res(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_ob(hidden_size, &SE);
    Buffer<uint8_t> attn_output_dropout_mask(batch_size * sequence_length * hidden_size, &SE); 
    Buffer<float> ff1_inp(batch_size * sequence_length * hidden_size, &SE);
    Buffer<float> attn_nw(hidden_size, &SE);
    Buffer<float> attn_nb(hidden_size, &SE);
    Buffer<float> attn_norm_var(batch_size * sequence_length, &SE);
    Buffer<float> attn_norm_mean(batch_size * sequence_length, &SE);
    Buffer<float> inter_w(hidden_size * intermediate_size, &SE);
    Buffer<float> inter_b(intermediate_size, &SE);
    Buffer<float> ff2_inp(batch_size * sequence_length * intermediate_size, &SE);
    Buffer<float> output_w(intermediate_size * hidden_size, &SE);
    Buffer<float> output_b(hidden_size, &SE);
    Buffer<float> out(batch_size * sequence_length * hidden_size, &SE);
    Buffer<uint8_t> layer_output_dropout_mask(batch_size * sequence_length * hidden_size, &SE); 

    _layer_norm.SetMeansAndVariance(&norm_mean, &norm_var);
    _attn_layer_norm.SetMeansAndVariance(&attn_norm_mean, &attn_norm_var);

    //printf("\x1b[31;1mExecuting layer norm\x1b[0m\n");
    sw.start();
    _layer_norm.ForwardCheckpoint(bsz, 
                                 &input_norm, 
                                 &input, 
                                 &norm_weights, 
                                 &norm_bias, 
                                 &SE, 
                                 sync);
    sw.stop();
    printf("\x1b[32;1mExecuted layer norm in %fs\x1b[0m\n", sw.GetTimeInSeconds());
    printf("layer_norm:%f\n", sw.GetTimeInSeconds());

    //printf("\x1b[31;1mExecuting qkv_linear\x1b[0m\n");
    sw.restart();
    _qkv_linear.ForwardCheckpointPartition(bsz,
                                           &input_norm,
                                           &qkv_weights,
                                           &qkv_out,
                                           &SE,
                                           nq,
                                           sync);
    sw.stop();
    printf("qkv_linear:%f\n", sw.GetTimeInSeconds());
    
    //printf("\x1b[31;1mExecuting launch_bias_add_transform_0213\x1b[0m\n");
    sw.restart();
    launch_bias_add_transform_0213<float>(queries.get_device_data(), 
                                          qkv_out.get_device_data(), 
                                          qkv_bias.get_device_data(), 
                                          batch_size, 
                                          sequence_length, 
                                          hidden_size, 
                                          nh, 
                                          SE.getStream(0), 
                                          3);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("launch_bias_add_transform_0213:%f\n", sw.GetTimeInSeconds());
    
    //printf("\x1b[31;1mCalculating attention scores\x1b[0m\n");
    sw.restart();
    _attn_scores.Forward(bsz_heads, &soft_out, &keys, &queries, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("attention_scores:%f\n", sw.GetTimeInSeconds());
    
    //printf("\x1b[31;1mExecuting softmax\x1b[0m\n");
    sw.restart();
    _softmax.ForwardCheckpoint(batch_size, &soft_out, &input_mask, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("softmax:%f\n", sw.GetTimeInSeconds());

    _attn_prob_dropout.SetMask(&attn_prob_dropout_mask);
    //printf("\x1b[31;1mExecuting attention probability dropout\x1b[0m\n");
    sw.restart();
    _attn_prob_dropout.Forward(bsz_heads * sequence_length, &context, &soft_out, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("attention_probability_dropout:%f\n", sw.GetTimeInSeconds());

    //printf("\x1b[31;1mCalculating attention context\x1b[0m\n");
    sw.restart();
    _attn_context.Forward(bsz_heads, &buf_1, &values, &context, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("attention_context:%f\n", sw.GetTimeInSeconds());
    
    sw.restart();
    launch_transform4d_0213<float>(attn_o_inp.get_device_data(), 
                                   buf_1.get_device_data(), 
                                   batch_size, 
                                   nh, 
                                   sequence_length, 
                                   hidden_size, 
                                   SE.getStream(0), 
                                   1); 
                                  
    sw.stop();
    printf("launch_transform4d_0213:%f\n",sw.GetTimeInSeconds());
    //printf("\x1b[31;1mExecuting attention out\x1b[0m\n");
    sw.restart();
    _attn_out_linear.ForwardCheckpoint(bsz_seq, &attn_o_inp, &attn_ow, &buf_1, &SE, true);
    sw.stop();
    printf("attention_out_linear:%f\n", sw.GetTimeInSeconds());
    
    _attn_output_dropout.SetMask(&attn_output_dropout_mask);
    //printf("\x1b[31;1mExecuting attention output dropout\x1b[0m\n");
    sw.restart();
    _attn_output_dropout.ForwardWithBias(bsz_seq, &add_res, &buf_1, &input, &attn_ob, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("attention_output_dropout:%f\n", sw.GetTimeInSeconds());

    //printf("\x1b[31;1mExecuting attention layer norm\x1b[0m\n");
    sw.restart();
    _attn_layer_norm.ForwardCheckpoint(bsz_seq, &ff1_inp, &add_res, &attn_nw, &attn_nb, &SE, true);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("attention_layer_norm:%f\n", sw.GetTimeInSeconds());

    //printf("\x1b[31;1mExecuting 1st feed forward layer\x1b[0m\n");
    sw.restart();
    _ff1.ForwardCheckpoint(bsz_seq, &ff1_inp, &inter_w, &ff2_inp, &SE, true);
    sw.stop();
    printf("1st_feed_forward_layer:%f\n", sw.GetTimeInSeconds());

    //printf("\x1b[31;1mExecuting gelu\x1b[0m\n");
    sw.restart();
    _gelu.ForwardWithBiasAdd(bsz_seq, &ff2_inp, &inter_b, &context, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("gelu:%f\n", sw.GetTimeInSeconds());

    //printf("\x1b[31;1mExecuting 2nd feed forward layer\x1b[0m\n");
    sw.restart();
    _ff2.ForwardCheckpoint(bsz_seq, &context, &output_w, &out, &SE, true);
    sw.stop();
    printf("2nd_feed_forward_layer:%f\n", sw.GetTimeInSeconds());

    _layer_output_dropout.SetMask(&layer_output_dropout_mask);
    //printf("\x1b[31;1mExecuting layer output dropout\x1b[0m\n");
    sw.restart();
    _layer_output_dropout.ForwardWithBias(bsz_seq, &out, &out, &add_res, &output_b, &SE);
    CHECK(cudaThreadSynchronize());
    sw.stop();
    printf("layer_output_dropout:%f\n", sw.GetTimeInSeconds());
    
    return 0;
}
