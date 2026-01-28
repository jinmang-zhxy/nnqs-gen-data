#!/bin/bash
set +x

# Submit your job

# (1) Serial task
# eg. `./submit.sh ../molecules/lih/qubit_op.data config-lih.yaml lih-test`
# This will output into file `out.lih-test`
# Your can modify the submit.sh acoording your compute platform
# ./submit.sh <hamiltonian_path> <config_file_name> <out_file_name_suffix>

# (2) Parallel task
# eg. `./submit.sh ../molecules/lih/qubit_op.data config-lih.yaml lih-test-p2 parallel`
# This will output into file `out.lih-test-p2`
# Your can modify the submit.sh acoording your compute platform
# ./submit.sh <hamiltonian_path> <config_file_name> <out_file_name_suffix> parallel

# (3) Foundation training serial task
# 3.1 set train_type="foundation" in this file
# 3.2 ./submit.sh h2_ham configs/config-h2.yaml h2-foundation
# ./submit.sh <anything_placeholder> <config_file_name> <out_file_name_suffix>

export LANG=en

hamiltonian_path=$1
config_file_name=$2
out_file_name_suffix=$3
task_type=$4 # optional

#=== user-defined parameters begin ===
# cuda_visible_devices="0,1,2,3" # parallel task examples
cuda_visible_devices="0,1,2,3,4,5,6,7" # parallel task examples
cuda_visible_devices=0,1,2,3 # serial task examples
# job_type="slurm" # slurm system
job_type="local" # local system
train_type="base" # for normal ab-initio train
# train_type="foundation" # for foundation model train

# Only used for parallel training, modify them according your requirements.
MASTER_PORT=8890 # ATTENTION: submit simultaneously multiple jobs should change this port!
NUM_NODES=1
NUM_PROC_PER_NODE=2

# when NUM_NODES>1, config follows:
RDZV_ID=wyj111
RDZV_BACKEND=c10d
RDZV_ENDPOINT=node-1:13345 # your master node's ip:port
#=== user-defined parameters end ===


out_file=out.${out_file_name_suffix}
exec > ${out_file} 2>&1 
which python
nvcc --version

commit_id=`git log | head -n 1`
echo git commit id: $commit_id

echo hamiltonian_path: ${hamiltonian_path}
echo config_file_name: ${config_file_name}
echo cuda_visible_devices: ${cuda_visible_devices}
echo ">>>Configuration Begin<<<"
grep "#" ${config_file_name} -rv
echo ">>>Configuration End<<<"
echo task_type: ${task_type}
if [[ -n ${task_type} && ${task_type} == "parallel" ]]; then
	if [ ${train_type} == "base" ]; then
        # e.g. torchrun --nnodes=1 --nproc_per_node=3 nnqs.py ../molecules/thomas/h2o/qubit_op.data configs/config-h2o.yaml --log_file=out.h2o-p3 --parallel
        python_cmd="nnqs.py ${hamiltonian_path} ${config_file_name} --log_file=${out_file}"
	else
        # e.g. torchrun --nnodes=1 --nproc_per_node=4 --master-port=8890 pretrain.py configs/config-h2.yaml --log_file=out.h2-foundation-parallel --use_parallel
        python_cmd="pretrain.py ${config_file_name} --log_file=${out_file}"
	fi

    if [ "$NUM_NODES" -gt 1 ]; then
        torchrun_cmd="torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_PROC_PER_NODE} --master-port=${MASTER_PORT} --rdzv-id=${RDZV_ID} --rdzv-backend=${RDZV_BACKEND} --rdzv-endpoint=${RDZV_ENDPOINT}"
    else
        torchrun_cmd="torchrun --nnodes=${NUM_NODES} --nproc_per_node=${NUM_PROC_PER_NODE} --master-port=${MASTER_PORT}"
    fi
    exec_cmd="${torchrun_cmd} ${python_cmd} --use_parallel"
else
	if [ ${train_type} == "base" ]; then
    	exec_cmd="python nnqs.py ${hamiltonian_path} ${config_file_name} --log_file=${out_file}"
	else
    	exec_cmd="python pretrain.py ${config_file_name} --log_file=${out_file}" # foundation model training
	fi
fi

if [ "$job_type" == "slurm" ]; then
    # slurm job system
    # ./job_script/gen.sh ${out_file} job_script/job-gpu-tp.sh
    bash job_script/gen.sh ${out_file} job_script/job-gpu-tp.sh "${exec_cmd}"
    sbatch _jobcp.sh
    rm _jobcp.sh
else
    # local system
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nohup ${exec_cmd} >> ${out_file} 2>&1 &
fi
chmod o+w ${out_file}

exec > /dev/tty 2>&1  
echo "log file is ${out_file}"
echo "train_type: ${train_type}"
echo $exec_cmd
