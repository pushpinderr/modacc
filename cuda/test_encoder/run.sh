./encoder $1 1024 1024 4096 16 4 > encoder_stats.txt
python run.py ./fine_grained_normalize $1 1024 1024
python run.py ./fine_grained_gemm $1 1024 1024
python run.py ./fine_grained_ff1 $1 1024 1024 4096 16
python run.py ./fine_grained_ff2 $1 1024 1024 4096 16
