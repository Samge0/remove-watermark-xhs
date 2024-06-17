#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-05-31 16:04
# describe：使用yolo8+iopaint结合，用yolo8识别目标水印位置，调用iopaint移除水印

import cv2
import os
from ultralytics.utils import checks
from yolo_utils import YOLOUtils
from iopaint_utils import IOPaintCmdUtil, IOPaintApiUtil

CUDA_IS_AVAILABLE = checks.cuda_is_available()

output_dir = ".cache"                               # 输出目录
model_path = "models/best.pt"                       # yolo模型路径
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"     # 设备类型

USE_IOPAINT_API = True                              # 【推荐】是否使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务


# 擦除水印
def detect_and_erase(image_path, model_path, output_dir, device="cpu"):
    # 初始化YOLO模型和IOPaint工具
    yolo_obj = YOLOUtils(model_path)
    iopaint_obj = IOPaintApiUtil(device=device) if USE_IOPAINT_API else IOPaintCmdUtil(device=device)

    # 读取图像
    image = cv2.imread(image_path)

    # 使用YOLO模型获取边界框
    bboxes = yolo_obj.get_bboxes(image)

    # 创建并保存掩码图像
    mask = iopaint_obj.create_mask(image, bboxes)
    iopaint_obj.erase_watermark(image_path, mask, output_dir)


# Run batch
def _test_batch(batch_dir: str):
    file_list = os.listdir(batch_dir)
    total_size = len(file_list)
    for i in range(total_size):
        filename = file_list[i]

        is_goal_image = filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        if not is_goal_image:
            continue

        image_path = os.path.join(batch_dir, filename).replace(os.sep, "/")
        print(f"【{i+1}/{total_size}】 is running: {image_path} => {output_dir}")
        detect_and_erase(image_path, model_path, output_dir, device=device)


if __name__ == "__main__":
    """
    使用示例
    """

    if USE_IOPAINT_API:
        print("=====【温馨提示】使用iopaint的api方式去除水印，如果设置为True，需要先运行iopaint服务：python iopaint_server.py 或使用自定义的IOPaint服务=====\n")

    os.makedirs(output_dir, exist_ok=True)

    # 移除单张水印
    image_path = "resources/test.jpg"
    detect_and_erase(image_path, model_path, output_dir, device=device)

    # 移除某个目录所有图片水印
    _test_batch(batch_dir="resources")

    print("all done")
