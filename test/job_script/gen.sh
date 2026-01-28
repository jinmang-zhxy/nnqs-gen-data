#!/bin/bash
OUTFILE=$1
JOB_NAME=$OUTFILE
#QUEUE_NAME=GPU-A100-80G
QUEUE_NAME=GPU-A100
NODE_NUM=1
TASK_NUM=1
NTASK_PER_NODE=1
NTASK_PER_SOCKET=8
TPF=_jobcp.sh
job_tp_file=$2
cp ${job_tp_file} $TPF
sed -i 's/JOB_NAME/'$JOB_NAME'/' $TPF 
sed -i 's/QUEUE_NAME/'$QUEUE_NAME'/' $TPF 
# sed -i 's/OUTFILE/'$OUTFILE'/g' $TPF 
sed -i 's/NODE_NUM/'$NODE_NUM'/' $TPF 
sed -i 's/TASK_NUM/'$TASK_NUM'/' $TPF 
sed -i 's/NTASK_PER_SOCKET/'$NTASK_PER_SOCKET'/' $TPF 
sed -i 's/NTASK_PER_NODE/'$NTASK_PER_NODE'/' $TPF

echo "${exec_cmd} >> ${OUTFILE} 2>&1" >> $TPF
echo "write job configurations into "$TPF
#cat $TPF
