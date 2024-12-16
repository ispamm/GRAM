config_name='pretrain_vast'
output_dir=./output/gram/$config_name

python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9834 \
./run.py \
--config ./config/gram/pretrain/$config_name.json \ 
--output_dir $output_dir \
--checkpointing true 

 