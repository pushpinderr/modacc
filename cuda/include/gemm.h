#include "common.h"
struct GemmConfig
{
    GemmConfig(cublasOperation_t trans_A, cublasOperation_t trans_B, int batch, int sequence_length, int hidden_size, float a, float b)
    {
        
        
        op_A = trans_A;
        op_B =  trans_B;
        

        int n = batch * sequence_length;
        int k = hidden_size;
        int m = 3*hidden_size;

        alpha = a;
        beta = b;
        lda = (op_A == CUBLAS_OP_N)? m:k;
        ldb = (op_B == CUBLAS_OP_N)? k:n;
        ldc = m;
        
        this->m = m;
        this->n = n;
        this->k = k;
        this->batch = batch;        
    }


    cublasOperation_t op_A;
    cublasOperation_t op_B;

    float alpha;
    float beta;
    long long int lda,ldb,ldc;

    int m, n, k, batch;

};