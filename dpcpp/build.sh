rm build_test.sh.*
qsub -l nodes=s012-n004:iris_xe_max:ppn=2 -d . scripts/build_test.sh
watch -n 1 qstat -n -1

