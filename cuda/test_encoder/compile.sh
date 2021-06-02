nvcc -std=c++11 encoder.cu -lcublas -o encoder
nvcc -w -std=c++11 fine_grained_gemm.cu -lcublas -o fine_grained_gemm
nvcc -std=c++11 fine_grained_normalize.cu -lcublas -o fine_grained_normalize
nvcc -std=c++11 fine_grained_attention_output_linear.cu -lcublas -o fine_grained_attention_output_linear
nvcc -std=c++11 fine_grained_ff1.cu -lcublas -o fine_grained_ff1
nvcc -std=c++11 fine_grained_ff2.cu -lcublas -o fine_grained_ff2
nvcc -std=c++11 -DEVENT_PROFILE=1 profile_encoder.cu -lcublas -o profile_encoder
