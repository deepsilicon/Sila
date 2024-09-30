#!/bin/bash

sbatch --wait -o replit_train_resume_9k.out <<EOF
#!/bin/bash
#SBATCH --job-name=replit-train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00

# Set ENROOT_RUNTIME_PATH to use /fsx
export ENROOT_RUNTIME_PATH=/fsx/\$USER/enroot_runtime

# Ensure the directory exists
mkdir -p \$ENROOT_RUNTIME_PATH

# Get the list of allocated nodes
NODES=(\$(sinfo -hN|awk '{print \$1}'))
MASTER_NODE=\${NODES[0]}
MASTER_ADDR=\$(srun --nodes=1 --ntasks=1 -w \$MASTER_NODE hostname -i | head -n 1)
MASTER_PORT=7501

echo "Job ID: \$SLURM_JOB_ID"
echo "Node List: \${NODES[@]}"
echo "Master Node: \$MASTER_NODE"
echo "Master Address: \$MASTER_ADDR"

# Launch master node
srun --container-image /fsx/desicr+replitrun_perservere_rms+latest.sqsh --container-workdir /bit-replit/scripts/train \
    --ntasks=1 --nodes=1 -w \$MASTER_NODE \
    bash -c "export AWS_ACCESS_KEY_ID=<AWS KEY> && \
    export AWS_SECRET_ACCESS_KEY=<AWS SECRET> && \
    export AWS_DEFAULT_REGION=us-east-2 && \
    export WANDB_API_KEY=<WANDBKEY> && \
    python clean_mem_distrib.py && \
    export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
    echo \"Running on master node \$MASTER_NODE with rank 0\" && \
    composer --world_size 64 --node_rank 0 \
    --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT \
    train.py yamls/pretrain/ternary-replit-3b-star-27-raw.yaml train_loader.dataset.split=train" &

# Wait for a few seconds to ensure master node is initialized
sleep 5

# Launch worker nodes in parallel
srun --container-image /fsx/desicr+replitrun_perservere_rms+latest.sqsh --container-workdir /bit-replit/scripts/train \
    --ntasks=7 --nodes=7 --exclude=\$MASTER_NODE \
    bash -c "export AWS_ACCESS_KEY_ID=<AWS KEY> && \
    export AWS_SECRET_ACCESS_KEY=<AWS SECRET> && \
    export AWS_DEFAULT_REGION=us-east-2 && \ 
    export WANDB_API_KEY=<WANDB KEY> && \ 
    export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
    node_rank=\\\$((\\\$SLURM_PROCID + 1)) && \
    echo \"Running on worker node \\\$SLURMD_NODENAME with rank \\\$node_rank\" && \
    composer --world_size 64 --node_rank \\\$node_rank \
    --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT \
    train.py yamls/pretrain/ternary-replit-3b-star-27-raw.yaml train_loader.dataset.split=train" &

wait
echo "All tasks completed"
