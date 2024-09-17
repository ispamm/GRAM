import os
import tqdm
import ffmpeg
import subprocess
import multiprocessing
import numpy as np

from multiprocessing import Pool


input_path = '/public/chensihan/datasets/tgif/gifs_used'
output_path = '/public/chensihan/datasets/tgif/'

data_list = os.listdir(input_path)

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipline(video_path, video_probe, output_dir, fps, sr, duration_target):
    video_name = os.path.basename(video_path)

    video_name = video_name.replace(".mp4", "")


    # extract video frames fps
    fps_frame_dir = os.path.join(output_dir, f"frames_fps{fps}", video_name)
    os.makedirs(fps_frame_dir, exist_ok=True)
    cmd = "ffmpeg -loglevel error -i {} -vsync 0 -f image2 -vf fps=fps={:.02f} -qscale:v 2 {}/frame_%04d.jpg".format(
              video_path, fps, fps_frame_dir)

    ## extract fixed number frames
    # fps_frame_dir = os.path.join(output_dir, f"frames_32", video_name)
    # os.makedirs(fps_frame_dir, exist_ok=True)
    # cmd = "ffmpeg -loglevel error -i {} -vsync 0 -f image2 -vframes 32  -qscale:v 2 {}/frame_%04d.jpg".format(
    #           video_path,  fps_frame_dir)
  

    # ## extract audios
    # sr_audio_dir = os.path.join(output_dir,f"audios")
    # os.makedirs(sr_audio_dir, exist_ok=True)
    # # print(sr_audio_dir)
    # audio_name = video_name+'.wav'
    # audio_file_path = os.path.join(sr_audio_dir, audio_name)


    cmd = "ffmpeg -i {} -loglevel error -f wav -vn -ac 1 -ab 16k -ar {} -y {}".format(
            video_path, sr, audio_file_path)
  

    subprocess.call(cmd, shell=True)



def extract_thread(video_id):
    
    video_name = os.path.join(input_path, video_id)
    
    if not os.path.exists(video_name):
     
        return
    try:
        # print(1)
        probe = ffmpeg.probe(video_name)
        # print(1)
        pipline(video_name, probe, output_path, fps=1, sr=22050, duration_target=10)
    except Exception as e:
        print(e)
        return 


def extract_all(video_ids, thread_num, start):
    length = len(video_ids)
    print(length)
    with Pool(thread_num) as p:
        list(tqdm.tqdm(p.imap(extract_thread, video_ids), total=length))

if __name__=='__main__':
    thread_num = 20
    start = 0

    print(len(data_list))
    extract_all(data_list, thread_num, start)

