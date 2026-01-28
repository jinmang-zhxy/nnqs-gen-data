#!/bin/bash
set -x

# config your cluster nodes ip list
NODE_IPS=("node-1" "node-2" "node-3" "node-5" "node-6" "node-7" "node-12" "node-16" "node-17" "node-18")

# Total number of nodes
NUM_NODES=${#NODE_IPS[@]} # len(NODE_IPS)
# NUM_NODES=8
GPUS_PER_NODE=4

# Master node information
MASTER_ADDR='node-1'
MASTER_PORT='19292'

# your conda path
CONDA="/home/nas/wuyangjun/miniconda3/bin/conda"
LOG_FILE="log-multinode"
SHARED_CONDA_DIR="/root/miniconda3/" # Path to the shared conda environment
# exec command
TRAIN_SCRIPT="pretrain.py configs/config-h2o2.yaml --log_file=out.h2o2-p2 --use_parallel"

# ln -s /home/nas/wuyangjun/miniconda3 /root/miniconda3
# Function to create symbolic link if it doesn't exist
create_symlink_if_not_exists() {
    local target=$1
    local link_name=$2
    if [ ! -L "$link_name" ]; then
        echo "Creating symbolic link: $link_name -> $target"
        sudo ln -s $target $link_name
    else
        echo "Symbolic link already exists: $link_name"
    fi
}

# Loop to start jobs on each node
for ((i=0; i<NUM_NODES; i++)); do
  # Calculate the rank of the current node
  RANK=$((i + 1))
  NODE=${NODE_IPS[i]}

  # Start the training process on the remote node via SSH
  echo "Starting job on $NODE with rank $i"
  ssh $NODE "bash -l -c '
    # Check and create the symbolic link if not exists
    $(typeset -f create_symlink_if_not_exists)
    create_symlink_if_not_exists /home/nas/wuyangjun/miniconda3 $SHARED_CONDA_DIR 

    # Initialize conda and activate the environment
    #source /root/.bashrc
    #export PATH=/home/nas/wuyangjun/miniconda3/bin:\$PATH
    source /home/nas/wuyangjun/miniconda3/etc/profile.d/conda.sh
    conda init bash
    source ~/.bashrc
    conda activate wyj_nnqs

    # Run the training script
    cd /home/nas/wuyangjun/NeuralNetworkQuantumState/
    source setup.sh
    cd test
    torchrun --nnodes=$NUM_NODES --node_rank=$i --nproc_per_node=$GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $TRAIN_SCRIPT
  '" >> "$LOG_FILE" 2>&1 &
done

wait  # Wait for all background processes to finish
exit
