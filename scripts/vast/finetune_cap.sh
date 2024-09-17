config_name='pretrain_vast'
output_dir=./output/vast/$config_name




##### VIDEO-CAP

# caption-msrvtt
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9634 \
./run.py \
--learning_rate 2e-5 \
--train_batch_size 128 \
--train_epoch 10 \
--checkpointing true \
--save_best true \
--config ./config/vast/finetune_cfg/caption-msrvtt.json \
--pretrain_dir $output_dir \
--beam_size 3 \
--first_eval false \
--output_dir  $output_dir/downstream/caption-msrvtt-tvas \

# # caption-msvd
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8  \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/caption-msvd.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir  $output_dir/downstream/caption-msvd \







# # caption-youcook
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 3e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/caption-youcook.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/caption-youcook \



# # caption-valor32k
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --pretrain_dir $output_dir \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/caption-valor32k.json \
# --output_dir $output_dir/downstream/caption-valor32k \
  
 
   


# # caption-tv
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 3e-5 \
# --checkpointing true \
# --train_epoch 40 \
# --save_best true \
# --config ./config/vast/finetune_cfg/caption-tv.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/caption-tv \


# # caption-vatex
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/caption-vatex.json \
# --pretrain_dir $output_dir \
# --output_dir  $output_dir/downstream/caption-vatex \
# --save_best true \


# #### AUDIO-CAP


# # caption-clothov2
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/caption-clothov2.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/caption-clothov2


# # caption-audiocaps
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/caption-audiocaps.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/caption-audiocaps


# #### IMAGE-CAP

# # caption-mscoco
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --config ./config/vast/finetune_cfg/caption-mscoco.json \
# --pretrain_dir $output_dir \
# --vision_resolution 480 \
# --output_dir $output_dir/downstream/caption-mscoco \
# --checkpointing false \
# --save_best true \

