python3 -m torch.distributed.launch \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 8 \
--master_port 9814 \
./run.py \
--config ./config/vast/captioner_cfg/caption-generation-vision.json \
--pretrain_dir './output/vast/vision_captioner' \
--output_dir './output/vast/vision_captioner/generation' \
--test_batch_size 64 \
--test_vision_sample_num 8 \
--generate_nums 3 \
--captioner_mode true \