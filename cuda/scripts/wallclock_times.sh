#arguments: batch_size, sequence_length, hidden_size(size of Q/K/V)
rm ./logs/wallclock_gemm_$1_$2_$3.txt
./obj/profile_wallclock_gemm $1 $2 $3 2 >> ./logs/wallclock_gemm_$1_$2_$3.txt
./obj/profile_wallclock_gemm $1 $2 $3 4 >> ./logs/wallclock_gemm_$1_$2_$3.txt
./obj/profile_wallclock_gemm $1 $2 $3 8 >> ./logs/wallclock_gemm_$1_$2_$3.txt
./obj/profile_wallclock_gemm $1 $2 $3 12 >> ./logs/wallclock_gemm_$1_$2_$3.txt
./obj/profile_wallclock_gemm $1 $2 $3 16 >> ./logs/wallclock_gemm_$1_$2_$3.txt
