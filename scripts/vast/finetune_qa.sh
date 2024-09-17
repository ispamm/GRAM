config_name='pretrain_vast'
output_dir=./output/vast/$config_name


### VIDEO-QA


# vqa-msrvtt
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node $ARNOLD_WORKER_GPU \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--beam_size 1 \
--config ./config/vast/finetune_cfg/VQA-msrvtt.json \
--first_eval false \
--save_best true \
--valid_freq 1 \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/VQA-msrvtt \



# vqa-msvd
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--config ./config/vast/finetune_cfg/VQA-msvd.json \
--pretrain_dir $output_dir \
--save_best true \
--output_dir $output_dir/downstream/VQA-msvd \






# vqa-tgif-frame
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--pretrain_dir $output_dir \
--config ./config/vast/finetune_cfg/VQA-tgif-frame.json \
--output_dir  $output_dir/downstream/VQA-tgif \






# vqa-anet
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--save_best true \
--config ./config/vast/finetune_cfg/VQA-activitynet.json \
--first_eval false \
--pretrain_dir $output_dir \
--output_dir $output_dir/downstream/VQA-activitynet \



# vqa-music-avqa
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--pretrain_dir $output_dir \
--first_eval false \
--checkpointing true \
--config ./config/vast/finetune_cfg/VQA-music.json \
--output_dir $output_dir/downstream/VQA-music




### IMAGE-QA


# vqa-vqav2
python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9824 \
./run.py \
--learning_rate 2e-5 \
--config ./config/vast/finetune_cfg/VQA-vqav2.json \
--pretrain_dir $output_dir \
--first_eval false \
--vision_resolution 384 \
--valid_freq 1 \
--output_dir $output_dir/downstream/VQA-vqav2 \



