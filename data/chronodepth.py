import argparse
import logging
import os
import random

from easydict import EasyDict
import numpy as np
import torch
from PIL import Image
# import mediapy as media
from tqdm.auto import tqdm

from chronodepth_pipeline import ChronoDepthPipeline

# def seed_all(seed: int = 0):
#     """
#     Set random seeds of all components.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def read_video(video_path):
#     return media.read_video(video_path)

# def export_to_video(video_frames, output_video_path, fps):
#     media.write_video(output_video_path, video_frames, fps=fps)
 # logging.basicConfig(level=logging.INFO)
cfg = EasyDict({
    "model_base": "jhshao/ChronoDepth",
    # "data_dir": "data/youcook2/youcook2.mp4",
    # "output_dir": "output",
    "denoise_steps": 10,
    "half_precision": True,
    # "num_frames": 8,
    "decode_chunk_size": 1,
    "window_size": 9,
    "seed": 1234,
    "inpaint_inference": False
    
})

# if cfg.window_size is None or cfg.window_size == cfg.num_frames:
#     cfg.inpaint_inference = False
# else:
#     cfg.inpaint_inference = True

# print(cfg)

# -------------------- Preparation --------------------
# -------------------- Device --------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    logging.warning("CUDA is not available. Running on CPU will be slow.")
# logging.info(f"device = {device}")
# local_rank = int(os.environ['LOCAL_RANK'])
# print("LOCAL RANK", local_rank, type(local_rank))
# device = torch.device("cuda", local_rank)
# -------------------- Random Seed --------------------
# if cfg.seed is None:
#     import time
#     cfg.seed = int(time.time())
# seed_all(cfg.seed)

# generator = torch.Generator(device).manual_seed(cfg.seed)

# assert cfg.data_dir.endswith(".mp4"), "data_dir should be mp4 file."
# os.makedirs(cfg.output_dir, exist_ok=True)
# logging.info(f"output dir = {cfg.output_dir}")

# -------------------- Model --------------------
if cfg.half_precision:
    weight_dtype = torch.float16
    logging.info(f"Running with half precision ({weight_dtype}).")
else:
    weight_dtype = torch.float32

pipeline = ChronoDepthPipeline.from_pretrained(
                        cfg.model_base,
                        torch_dtype=weight_dtype,
                        # batch_size=64
                    )

pipe = pipeline.to(device)

try:
    pipeline.enable_xformers_memory_efficient_attention()
except ImportError:
    logging.debug("run without xformers")

def extract_depth(rgb_int, save_path):
    # Check if the path contains 'videos_test' or 'videos_train' and replace accordingly
    if 'videos_test' in save_path:
        save_path = save_path.replace('videos_test', 'depth_test')
    elif 'videos_train' in save_path:
        save_path = save_path.replace('videos_train', 'depth_train')
    elif 'videos' in save_path: 
        save_path = save_path.replace('videos', 'depth')  
    else:
        save_path = save_path
    # -------------------- data --------------------
    # data_ls = []
    # video_data = read_video(cfg.data_dir)
    # fps = video_data.metadata.fps
    # for i in tqdm(range(len(video_data)-cfg.num_frames+1)):
    #     is_first_clip = i == 0
    #     is_last_clip = i == len(video_data) - cfg.num_frames
    #     is_new_clip = (
    #         (cfg.inpaint_inference and i % cfg.window_size == 0)
    #         or (cfg.inpaint_inference == False and i % cfg.num_frames == 0)
    #     )
    #     if is_first_clip or is_last_clip or is_new_clip:
    #         data_ls.append(np.array(video_data[i: i+cfg.num_frames])) # [t, H, W, 3]

    # video_name = cfg.data_dir.split('/')[-1].split('.')[0]
    # video_length = len(video_data)
    
    depth_colored_pred = []
    depth_pred = []
    rgb_int_ls = []
    # -------------------- Inference and saving --------------------
    # rgb_int = []#frame che abbiamo già in ingressoo
    with torch.no_grad():
        rgb_int = rgb_int.astype(np.uint8)

        input_images = [Image.fromarray(rgb_int[i]) for i in range(rgb_int.shape[0])]
        # Predict depth
        # if iter == 0: # First clip:
        pipe_out = pipe(
            input_images,
            num_frames=len(input_images),
            num_inference_steps=cfg.denoise_steps,
            decode_chunk_size=cfg.decode_chunk_size,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.0,
            # generator=generator,
        )
        # else: # Separate inference
        #     pipe_out = pipeline(
        #         input_images,
        #         num_frames=cfg.num_frames,
        #         num_inference_steps=cfg.denoise_steps,
        #         decode_chunk_size=cfg.decode_chunk_size,
        #         motion_bucket_id=127,
        #         fps=7,
        #         noise_aug_strength=0.0,
        #         generator=generator,
        #     )

        depth_frames_pred = [pipe_out.depth_np[i] for i in range(len(input_images))]

        depth_frames_colored_pred = []
        for i in range(len(input_images)):
            depth_frame_colored_pred = np.array(pipe_out.depth_colored[i])
            depth_frames_colored_pred.append(depth_frame_colored_pred)
        depth_frames_colored_pred = np.stack(depth_frames_colored_pred, axis=0)

        depth_frames_pred = np.stack(depth_frames_pred, axis=0)
        depth_frames_pred_ts = torch.from_numpy(depth_frames_pred).to(device)
        depth_frames_pred_ts = depth_frames_pred_ts * 2 - 1

        if cfg.inpaint_inference == False:
            # if iter == len(data_ls) - 1:
                # last_window_size = cfg.num_frames if video_length%cfg.num_frames == 0 else video_length%cfg.num_frames
                # rgb_int_ls.append(rgb_int[-last_window_size:])
                # depth_colored_pred.append(depth_frames_colored_pred[-last_window_size:])
                # depth_pred.append(depth_frames_pred[-last_window_size:])
            # else:
                rgb_int_ls.append(rgb_int)
                depth_colored_pred.append(depth_frames_colored_pred)
                depth_pred.append(depth_frames_pred)

        # rgb_int_ls = np.concatenate(rgb_int_ls, axis=0)
        depth_colored_pred = np.concatenate(depth_colored_pred, axis=0)
        # depth_pred = np.concatenate(depth_pred, axis=0)
        # # Save images
        
        output_dir_video = save_path[:-4] #remove mp4
        np.save(f"{output_dir_video}_depth.npy", depth_colored_pred)
        # print(f"Depth saved at {output_dir_video}_depth.npy")
        # os.makedirs(output_dir_video, exist_ok=True)
        # rgb_dir = os.path.join(output_dir_video, "rgb")
        # depth_colored_dir = os.path.join(output_dir_video, "depth_colored")
        # depth_pred_dir = os.path.join(output_dir_video, "depth_pred")
        # os.makedirs(rgb_dir, exist_ok=True)
        # os.makedirs(depth_colored_dir, exist_ok=True)
        # os.makedirs(depth_pred_dir, exist_ok=True)
        # for i in tqdm(range(len(rgb_int_ls))):
            # Image.fromarray(rgb_int_ls[i]).save(os.path.join(rgb_dir, f"frame_{i:06d}.png"))     
            # Image.fromarray(depth_colored_pred[i]).save(os.path.join(depth_colored_dir, f"frame_{i:06d}.png"))
            # np.save(os.path.join(depth_pred_dir, f"frame_{i:06d}.npy"), depth_pred[i])

    # print(depth_colored_pred == np.load(f"{output_dir_video}_depth.npy"))
    return depth_colored_pred