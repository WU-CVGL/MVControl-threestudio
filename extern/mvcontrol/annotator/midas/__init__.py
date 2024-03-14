import cv2
import torch
import numpy as np
import PIL
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation


class MidasDetector:
    pretrained_model_name_or_path = "intel/dpt-hybrid-midas"
    def __init__(self):
        self.image_processor = DPTImageProcessor.from_pretrained(
            self.pretrained_model_name_or_path
        )
        self.depth_predictor = DPTForDepthEstimation.from_pretrained(
            self.pretrained_model_name_or_path, torch_dtype=torch.float16
        ).cuda()

    @torch.no_grad()
    def __call__(self, input_image, mask):
        H, W = mask.shape
        
        pixel_values = self.image_processor(
            Image.fromarray(input_image), return_tensors="pt"
        )["pixel_values"].to(torch.float16).cuda()
        depth = self.depth_predictor(pixel_values=pixel_values).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), (H, W), mode="bicubic", antialias=True
        ).squeeze()

        depth = depth.float().cpu().numpy()
        depth[~mask] = 0
        depth_map = depth - depth.min()
        depth_map = depth_map / depth_map.max()
        depth_map = (depth_map * 255).astype(np.uint8)

        # normal
        x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * np.pi * 2
        x[~mask] = 0
        y[~mask] = 0
        normal = np.stack([x, y, z], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal_map = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        normal_map[~mask] = 0

        return depth_map, normal_map