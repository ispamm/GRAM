config_name='pretrain_vast'
output_dir=./output/vast/$config_name



### VIDEO-RET

#retrieval-msrvtt
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/vast/finetune_cfg/finetune-area.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/finetune_area \
#--mode 'testing' \
#--checkpoint ./output/vast/pretrain_vast/downstream/retrieval-msrvtt/ckpt/best_ret%tv--msrvtt_ret_ret_itc_tv.pt



# #retrieval-vatex
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-vatex.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-vatex \



# #retrieval-valor32k
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --pretrain_dir $output_dir \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-valor32k.json \
# --output_dir $output_dir/downstream/retrieval-valor32k \




# #retrieval-lsmdc
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --first_eval false \
# --config ./config/vast/finetune_cfg/retrieval-lsmdc.json \   
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-lsmdc \


# #retrieval-youcook
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 3e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-youcook.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-youcook \



# #retrieval-didemo
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-didemo.json \
# --pretrain_dir $output_dir \
# --save_best true \
# --output_dir $output_dir/downstream/retrieval-didemo \


# #retrieval-activitynet
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-activitynet.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-activitynet \
# --save_best true \



# ### AUDIO-RET

# #retrieval-clothov2
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-clothov2.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-clothov2 \

# #retrieval-audiocaps
# python3 -m torch.distributed.launch \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 8 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 2e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-audiocaps.json \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-audiocaps




# ### IMAGE_RET 

# #retrieval-mscoco
# python3 -m torch.distributed.launch \
# --nproc_per_node 8 \
# --master_port 9134 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-mscoco.json \
# --first_eval true \
# --save_best true \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-mscoco \
# --vision_resolution 384 \


# #retrieval-flickr
# python3 -m torch.distributed.launch \
# --nproc_per_node 8 \
# --master_port 9134 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --config ./config/vast/finetune_cfg/retrieval-flickr.json \
# --first_eval true \
# --save_best true \
# --pretrain_dir $output_dir \
# --output_dir $output_dir/downstream/retrieval-flickr \
# --vision_resolution 384 \

