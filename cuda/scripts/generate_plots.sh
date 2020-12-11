#!/bin/bash
echo -e "Generating binaries"
make profile_wallclock_gemm
make profile_stream_gemm
echo -e "Generated binaries"
echo -e "Running scripts"
bash scripts/wallclock_times.sh $1 $2 $3
bash scripts/stream_times.sh $1 $2 $3
echo -e "Generating Plots"
python scripts/plot_speedups.py wallclock logs/wallclock_gemm_$1_$2_$3.txt
python scripts/plot_speedups.py stream logs/stream_gemm_$1_$2_$3.txt
