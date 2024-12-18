#!/bin/sh
#SBATCH -A IscrC_NeuroGen
#SBATCH -p boost_usr_prod
#SBATCH --time=23:00:00      
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1    
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=8      
#SBATCH --job-name=msrvtt

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo_scratch/fast/IscrC_NeuroGen/luigi/GRAM
export WANDB_MODE=offline
module load anaconda3
source activate gram

# config_name='pretrain_gram'
# output_dir=./output/gram/$config_name

### VIDEO-RET

#retrieval-msrvtt
srun python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/gram/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir /leonardo_scratch/fast/IscrC_GenOpt/giordano/VAST/output/vast/pretrain_vast/downstream/finetuneVolume256batchlossonlyvolume4Mod120k \
--output_dir ./downstream/finetune_msrvtt \