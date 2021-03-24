rm run_ce.sh.*
qsub -l nodes=s012-n004:iris_xe_max:ppn=2 -d . scripts/run_ce.sh
watch -n 1 qstat -n -1

