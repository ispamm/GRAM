#!/bin/sh
#SBATCH -A IscrC_GenOpt
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=activity

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_NeuroGen/luigi/GRAM
export WANDB_MODE=offline
module load anaconda3
source activate gram

# config_name='pretrain_gram'
# output_dir=./output/gram/$config_name

### VIDEO-RET

srun python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--config ./config/gram/pretrain_cfg/pretrain_activity.json \
--output_dir ./downstream/pretrain_activity \
--checkpointing true \
--first_eval true \
--save_best true \

 