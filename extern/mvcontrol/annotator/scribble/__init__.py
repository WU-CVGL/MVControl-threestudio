# Pidinet
# https://github.com/hellozhuo/pidinet
import cv2
import os
import torch
import numpy as np
from einops import rearrange
from .model import pidinet
from extern.mvcontrol.annotator.util import HWC3, safe_step, annotator_ckpts_path, nms


class PidiNetDetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
        modelpath = os.path.join(annotator_ckpts_path, "table5_pidinet.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = pidinet()
        self.netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath)['state_dict'].items()})
        self.netNetwork = self.netNetwork.cuda()
        self.netNetwork.eval()

    def __call__(self, input_image, safe=False):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(input_image).float().cuda()
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0][0]
        
class ScribbleDetector:
    def __init__(self):
        self.pidi_detector = PidiNetDetector()

    def __call__(self, input_image, H, W):
        detected_map = self.pidi_detector(input_image)
        detected_map = HWC3(detected_map)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = nms(detected_map, 127, 3.0)
        detected_map = cv2.GaussianBlur(detected_map, (0, 0), 1.5)
        detected_map[detected_map > 4] = 255
        detected_map[detected_map < 255] = 0
        detected_map = detected_map.astype(np.uint8)

        return detected_map