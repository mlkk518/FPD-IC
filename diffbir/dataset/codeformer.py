from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from pathlib import Path

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr, random_crop_dual_arr
from ..utils.common import instantiate_from_config
import os

class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        file_gt: str,
        file_lq: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        random_flag: bool,
        Target_training: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        Load_latent: bool = False,
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()

        self.file_gt = self.get_image_files(file_gt)
        self.file_lq_root = file_lq

        self.Target_training = Target_training
        self.Load_latent = Load_latent

        # if self.Target_training:
        #     self.file_lq = self.get_image_files(file_lq)
        # self.prompt_list = ["0005", "0025", "0067", "025"]

        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        self.random_flag = random_flag
        # self.prompt_case = prompt_case
        # assert prompt_case in  self.prompt_list
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def load_gt_lq_image(
        self, gt_path: str, lq_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_gt_bytes = None
        while image_gt_bytes is None:
            if max_retry == 0:
                return None
            image_gt_bytes = self.file_backend.get(gt_path)
            image_lq_bytes = self.file_backend.get(lq_path)
            max_retry -= 1
            if image_gt_bytes is None or image_lq_bytes is None:
                time.sleep(0.5)
        img_gt = Image.open(io.BytesIO(image_gt_bytes)).convert("RGB")
        img_lq = Image.open(io.BytesIO(image_lq_bytes)).convert("RGB")
        if self.crop_type != "none":
            if img_gt.height == self.out_size and img_gt.width == self.out_size:
                img_gt = np.array(img_gt)
                img_lq = np.array(img_lq)
            else:
                if self.crop_type == "center":
                    img_gt = center_crop_arr(img_gt, self.out_size)
                    img_lq = center_crop_arr(img_lq, self.out_size)
                elif self.crop_type == "random":
                    img_gt, img_lq = random_crop_dual_arr(img_gt, img_lq, self.out_size, min_crop_frac=0.7)
        else:
            if self.random_flag: ## 训练时，才进行检查
                assert img_gt.height == self.out_size and img_gt.width == self.out_size
            img_gt = np.array(img_gt)
            img_lq = np.array(img_lq)
        # hwc, rgb, 0,255, uint8
        return img_gt, img_lq

    def random_flip(self, image_gt, image_lq, horizontal_prob=0.5, vertical_prob=0.5):
        """
        对图像进行随机翻转（水平或垂直）。
        """
        if random.random() < horizontal_prob:  # 水平翻转
            image_gt = image_gt[:, ::-1, :]  # 水平翻转
            image_lq = image_lq[:, ::-1, :]  # 水平翻转
        if random.random() < vertical_prob:  # 垂直翻转
            image_gt = image_gt[::-1, :, :]  # 垂直翻转
            image_lq = image_lq[::-1, :, :]  # 垂直翻转
        return image_gt, image_lq


    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:

        if self.Target_training:
            # load gt image
            gt_path = self.file_gt[index]
            gt_filename = os.path.basename(gt_path)
            lq_path = os.path.join(self.file_lq_root, gt_filename)
            # print("gt_path filename:", gt_path)
            # print("lq_path filename:", lq_path)
            assert gt_path.split("/")[-1] == lq_path.split("/")[-1]

            # if self.random_flag: # 训练阶段
            #     prompt = random.choice(self.prompt_list)
            #     lq_path_parts = self.file_lq[index].split("/")
            #     lq_path_parts[-2] = prompt
            #     lq_path = "/".join(lq_path_parts)
            #     prompt = self.generate_prompt(prompt)
            #     # if np.random.uniform() < 0.2:
            #     #     prompt = ""
            # else:  ## 测试阶段
            #     prompt = self.prompt_case
            #     prompt = self.generate_prompt(prompt)
            img_gt, img_lq = self.load_gt_lq_image(gt_path, lq_path)
            # img_lq = self.load_gt_image(lq_path)
            # prompt = ""
            if self.random_flag:  # 训练阶段
                img_gt, img_lq = self.random_flip(img_gt, img_lq, horizontal_prob=0.5, vertical_prob=0.5)

            if self.Load_latent:
                z_path = self.get_latent_path_os(lq_path)
                z_latent = np.load(z_path)
                z_latent = z_latent.squeeze()  # 结果形状: [4, 64, 64]

                # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
            img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
            img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
            h, w, _ = img_gt.shape


            # BGR to RGB, [-1, 1]
            gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
            # BGR to RGB, [0, 1]
            lq = img_lq[..., ::-1].astype(np.float32)

            if self.Load_latent:
                return gt, lq, z_latent

            if not self.random_flag: ## 测试阶段
                img_name = lq_path.split("/")[-1].split('.')[0]
                return gt, lq, img_name

        else: ## 预训练
            gt_path = self.file_gt[index]
            img_gt = self.load_gt_image(gt_path)
            img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
            h, w, _ = img_gt.shape
            # ------------------------ generate lq image ------------------------ #
            # blur
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None,
            )
            img_lq = cv2.filter2D(img_gt, -1, kernel)
            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_lq = cv2.resize(
                img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
            )
            # noise
            if self.noise_range is not None:
                img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
            # # jpeg compression
            # if self.jpeg_range is not None:
            #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)

            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

            # BGR to RGB, [-1, 1]
            gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
            # BGR to RGB, [0, 1]
            lq = img_lq[..., ::-1].astype(np.float32)

        return gt, lq

    def generate_prompt(self, compression_rate: str) -> str:
        # 定义压缩率（字符串）与描述的映射字典
        lambda_to_prompt = {
            "025": "R1: Optimized for high quality, resulting in higher bitrate.",
            "0067": "R2: Balanced optimization, achieving moderate quality and bitrate.",
            "0025": "R3: Optimized for lower bitrate, resulting in lower quality.",
            "0005": "R4: Optimized for very low bitrate, resulting in very low quality.",
        }

        # lambda_to_prompt = {
        #     "025": "A: Optimized for high quality, resulting in higher bitrate (bpp).",
        #     "0067": "B: Balanced optimization, providing moderate quality and bitrate (bpp).",
        #     "0025": "C: Optimized for low bitrate, resulting in lower quality.",
        #     "0005": "D: Optimized for very low bitrate, resulting in very low quality.",
        # }

        if compression_rate in lambda_to_prompt:
            return lambda_to_prompt[compression_rate]
        else:
            # 将压缩率转换为浮点数，以便更清晰地描述
            try:
                rate = float(compression_rate)
                return f"This is an image with a compression rate of {rate:.4f}."
            except ValueError:
                return "This is an image with an unknown compression rate."
    def get_image_files(self, directory):
        # 定义图片文件扩展名
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # 遍历目录并统计图片文件
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)  [1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        return image_files

    def get_latent_path_os(self, lq_path):
        """通过os.path实现路径转换"""
        # 分解原始路径
        dirname = os.path.dirname(lq_path)  # 获取原目录路径
        filename = os.path.basename(lq_path)  # 获取带扩展名的文件名

        # 替换目录中的LQ为latents
        new_dir = dirname.replace("LQ", "latents")

        # 替换文件扩展名
        base_name = os.path.splitext(filename)[0]  # 去除原扩展名
        new_filename = f"{base_name}.npy"

        # 组合新路径
        z_path = os.path.join(new_dir, new_filename)
        return z_path

    def __len__(self) -> int:
        return len(self.file_gt)
