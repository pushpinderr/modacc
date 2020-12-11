#arguments: batch_size, sequence_length, hidden_size(size of Q/K/V)
rm ./logs/stream_gemm_$1_$2_$3.txt
./obj/profile_stream_gemm $1 $2 $3 1 >> ./logs/stream_gemm_$1_$2_$3.txt
./obj/profile_stream_gemm $1 $2 $3 2 >> ./logs/stream_gemm_$1_$2_$3.txt
./obj/profile_stream_gemm $1 $2 $3 4 >> ./logs/stream_gemm_$1_$2_$3.txt
./obj/profile_stream_gemm $1 $2 $3 8 >> ./logs/stream_gemm_$1_$2_$3.txt
./obj/profile_stream_gemm $1 $2 $3 12 >> ./logs/stream_gemm_$1_$2_$3.txt
./obj/profile_stream_gemm $1 $2 $3 16 >> ./logs/stream_gemm_$1_$2_$3.txt
