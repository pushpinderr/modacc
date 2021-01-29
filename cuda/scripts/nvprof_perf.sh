echo -e "Generating binaries"
make profile_perf_gemm_coarse
make profile_perf_gemm_fine
make profile_perf_gemm_fine_all
nvprof --analysis-metrics -o ./logs/perf_coarse_$1_$2_$3_$i.nvprof ./obj/profile_perf_gemm_coarse $1 $2 $3 1
for s in 2 4 8 12 16 32
do
	nvprof --analysis-metrics -o ./logs/perf_fine_$1_$2_$3_$s.nvprof ./obj/profile_perf_gemm_fine $1 $2 $3 $s
done 
for s in 2 4 8 12 16 32
do
	nvprof --analysis-metrics -o ./logs/perf_fine_$1_$2_$3_$s-complete.nvprof ./obj/profile_perf_gemm_fine_all $1 $2 $3 $s
done 
