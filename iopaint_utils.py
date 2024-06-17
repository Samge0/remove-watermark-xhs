#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-03 14:50
# describe：iopaint的工具类


import uuid
import cv2
import os
import subprocess
import numpy as np

from iopaint_api_utils import InpaintAPI


class BaseIOPaint:
    """
    iopaint的工具类 - 基类
    """
    def __init__(self, device="cpu"):
        self.device = device

    def create_mask(self, image, bboxes, padding=1):
        height, width = image.shape[:2]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for bbox in bboxes:
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
            else:
                x1, y1, x2, y2, conf, cls = map(int, bbox)
            x1 = np.clip(x1 - padding, 0, width)
            y1 = np.clip(y1 - padding, 0, height)
            x2 = np.clip(x2 + padding, 0, width)
            y2 = np.clip(y2 + padding, 0, height)
            mask[y1:y2, x1:x2] = 255
        return mask

    def erase_watermark(self, image_path, mask, output_dir):
        pass


class IOPaintCmdUtil(BaseIOPaint):
    """
    命令行方式运行iopaint的工具类
    """

    def erase_watermark(self, image_path, mask, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        temp_mask_path = f'.cache/{uuid.uuid4()}{image_name}'
        cv2.imwrite(temp_mask_path, mask)

        command = [
            'iopaint', 'run',
            '--model=lama',
            f'--device={self.device}',
            f'--image={image_path}',
            f'--mask={temp_mask_path}',
            f'--output={output_dir}'
        ]

        try:
            subprocess.run(command, check=True)
            output_path = f"{output_dir}/{image_name}"
            print(f"水印已移除： {image_path} => {output_path}")
        finally:
            os.remove(temp_mask_path) if os.path.exists(temp_mask_path) else None

    def erase_watermark_batch(self, image_dir, mask_dir, output_dir):
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        output_dir = output_dir or output_dir.mkdtemp()
        os.makedirs(output_dir, exist_ok=True)

        command = [
            'iopaint', 'run',
            '--model=lama',
            f'--device={self.device}',
            f'--image={image_dir}',
            f'--mask={mask_dir}',
            f'--output={output_dir}'
        ]

        try:
            subprocess.run(command, check=True)
            print(f"水印已移除： => {output_dir}")
        finally:
            pass


class IOPaintApiUtil(BaseIOPaint):
    """
    调用api方式运行iopaint的工具类
    """

    def erase_watermark(self, image_path, mask, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        image_name = os.path.basename(image_path)
        temp_mask_path = f'.cache/{uuid.uuid4()}{image_name}'
        cv2.imwrite(temp_mask_path, mask)

        # 创建InpaintAPI类的实例 + 发送请求
        output_path = f"{output_dir}/{image_name}"
        inpaint_api = InpaintAPI()
        try:
            inpaint_api.send_inpaint_request(image_path, temp_mask_path, output_path)
            print(f"水印已移除： {image_path} => {output_path}")
        finally:
            os.remove(temp_mask_path) if os.path.exists(temp_mask_path) else None
