config_name='pretrain_gram'
output_dir=./output/gram/$config_name



### VIDEO-RET

#pretrain GRAM on VAST27M
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/gram/finetune_cfg/pretrain-gram.json \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/pretrain \
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

