#!/bin/sh

#SBATCH -J JOB_NAME
#SBATCH -p QUEUE_NAME
##SBATCH -o OUTFILE
##SBATCH -e OUTFILE
#SBATCH -N NODE_NUM
#SBATCH -n TASK_NUM
#SBATCH --ntasks-per-node=NTASK_PER_NODE
#SBATCH --cpus-per-task=1
##SBATCH --ntasks-per-socket=NTASK_PER_SOCKET
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:a100:NTASK_PER_NODE
##SBATCH --gpus-per-task=1
##SBATCH  --gpus-per-node=4
##SBATCH --qos=qos_a100_gpu

#export OPENBLAS_NUM_THREADS=6
export LANG=en

# which python
#module load mpi/hpcx/2.6.0/hpcx-intel-2020
#module load cuda/11.7
#module load nvhpc/21.11

# exec_cmd=$1
# out_file=OUTFILE
# ${exec_cmd} >> ${out_file} 2>&1
