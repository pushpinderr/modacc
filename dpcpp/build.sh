rm build_matmul.sh.*
qsub -l nodes=s012-n003:iris_xe_max:ppn=2 -d . scripts/build_matmul.sh
watch -n 1 qstat -n -1

