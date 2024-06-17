#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-17 11:05
# describe：

import shutil
import time
import cv2
import os
import tempfile

import torch
import ffmpeg
from iopaint_utils import IOPaintCmdUtil
from yolo_utils import YOLOUtils


CUDA_IS_AVAILABLE = torch.cuda.is_available()

output_dir = ".cache"                               # 输出目录
model_path = "models/best.pt"                       # yolo模型路径
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"     # 设备类型


# 读取视频信息
def get_video_info(video_path):
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    return int(video_info['width']), int(video_info['height']), int(video_info['r_frame_rate'].split('/')[0])


# 将视频帧合并为新视频
def create_video_from_images(image_folder, output_video_path, width, height, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()

    if not images:
        print("No images found in the directory.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        frame_resized = cv2.resize(frame, (width, height))
        video.write(frame_resized)

    video.release()
    print(f"Silent video saved at {output_video_path}")


# 创建新视频，合并原始音频
def create_video_with_audio(image_folder, audio_video_path, output_path):
    width, height, fps = get_video_info(audio_video_path)
    # 创建无声视频
    silent_video_path = f"{output_dir}/result_{int(time.time())}_silent_{os.path.basename(audio_video_path)}"
    create_video_from_images(image_folder, silent_video_path, width, height, fps)

    # 将音频合并到视频
    input_video = ffmpeg.input(silent_video_path)
    input_audio = ffmpeg.input(audio_video_path)
    ffmpeg.output(input_video.video, input_audio.audio, output_path, vcodec='copy', acodec='aac', strict='experimental').run()
    print(f"Video with audio saved at {output_path}")


# 提取视频的所有帧
def extract_frames(video_path, temp_dir = None):
    temp_dir = temp_dir or tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return None
    
    frame_count = 0
    success, image = video_capture.read()
    
    while success:
        # Save frame as JPEG file
        frame_filename = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, image)
        
        # Read next frame
        success, image = video_capture.read()
        frame_count += 1
    
    video_capture.release()
    print(f"Extracted {frame_count} frames to {temp_dir}")
    return temp_dir


# 根据帧图集合目录，创建一一对应的mask目录
def create_mask_dir(image_dir, temp_dir = None):
    temp_dir = temp_dir or tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    with os.scandir(image_dir) as entries:
        for entry  in entries:
            # 读取图像
            image = cv2.imread(entry.path)

            # 使用YOLO模型获取边界框
            bboxes = yolo_obj.get_bboxes(image)

            # 创建并保存掩码图像
            mask = iopaint_obj.create_mask(image, bboxes)
            image_name = entry.name
            temp_mask_path = f'{temp_dir}/{image_name}'
            cv2.imwrite(temp_mask_path, mask)
    return temp_dir


if __name__ == "__main__":
    # Example usage
    
    yolo_obj = YOLOUtils(model_path)
    iopaint_obj = IOPaintCmdUtil(device=device)
    
    # 读取所有视频帧到临时目录
    print(">"*40, " 读取所有视频帧到临时目录 ", ">"*40)
    start_time_0 = time.time()
    video_path = 'resources/test.mp4'
    temp_image_dir = extract_frames(video_path)
    print(f"Frames are saved in: {temp_image_dir}")
    print(f"read frames：{time.time() - start_time_0}s")
    
    # YOLO识别并创建帧图集对应的遮罩图集目录
    print(">"*40, " YOLO识别并创建帧图集对应的遮罩图集目录 ", ">"*40)
    start_time = time.time()
    temp_mask_dir = create_mask_dir(temp_image_dir)
    print(f"Frames are saved in: {temp_mask_dir}")
    print(f"create mask：{time.time() - start_time}s")
    
    # IOPaint批量移除水印
    print(">"*40, " IOPaint批量移除水印 ", ">"*40)
    start_time = time.time()
    temp_output_dir = tempfile.mkdtemp()
    iopaint_obj.erase_watermark_batch(temp_image_dir, temp_mask_dir, temp_output_dir)
    print(f"temp_output_dir in: {temp_mask_dir}")
    print(f"IOPaint批量移除水印：{time.time() - start_time}s")
    
    # 合并音视频
    print(">"*40, " 合并音视频 ", ">"*40)
    start_time = time.time()
    image_folder = temp_output_dir
    output_video_path = f"{output_dir}/result_{int(time.time())}_{os.path.basename(video_path)}"
    create_video_with_audio(image_folder, video_path, output_video_path)
    print(f"合并音视频：{time.time() - start_time}s")
    
    # 移除临时文件
    print(">"*40, " 移除临时文件 ", ">"*40)
    shutil.rmtree(temp_image_dir)
    shutil.rmtree(temp_mask_dir)
    shutil.rmtree(temp_output_dir)
    
    print(f"全部操作总耗时：{time.time() - start_time_0}s")
    print("all done")
