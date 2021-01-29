#arguments: batch_size, sequence_length, hidden_size(size of Q/K/V)
rm ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 2 >> ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 4 >> ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 8 >> ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 12 >> ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 16 >> ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 24 >> ./logs/reorder_$1_$2_$3.txt
./obj/profile_reorder $1 $2 $3 32 >> ./logs/reorder_$1_$2_$3.txt
