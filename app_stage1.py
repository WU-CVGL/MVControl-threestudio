import os
import cv2
import glob
import tyro
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import load_file
import rembg
import gradio as gr

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from extern.lgm.options import AllConfigs, Options
from extern.lgm.models import LGM
from extern.mvcontrol.pipeline_mvcontrol import load_mvcontrol_pipeline
from extern.mvcontrol.utils.camera import get_camera
from extern.mvcontrol.annotator import CannyDetector, MidasDetector, ScribbleDetector

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
GRADIO_VIDEO_PATH = 'gradio_output.mp4'
GRADIO_PLY_PATH = 'gradio_output.ply'

opt = tyro.cli(AllConfigs)

pretrained_model_name_or_path = "lzq49/mvdream-sd21-diffusers"
pretrained_controlnet_name_or_path = f"lzq49/mvcontrol-4v-{opt.condition_type}"

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

pipe_mvcontrol = load_mvcontrol_pipeline(
    pretrained_model_name_or_path, 
    pretrained_controlnet_name_or_path,
    num_views=4,
    weights_dtype=torch.float16,
    device=device
).to(device)

def pad_to_square(img, bg_color="white"):
    H, W, C = img.shape
    size = max(H, W)

    if bg_color == "white" and C == 3:
        new_img = (np.ones((size, size, C)) * 255).astype(np.uint8)
    else:
        new_img = np.zeros((size, size, C), dtype=np.uint8)

    if size == H:
        margin = (size - W) // 2
        new_img[:, margin:margin+W] = img
    else:
        margin = (size - H) // 2
        new_img[margin:margin+H, :] = img
    return new_img


# annotators
def get_annotator(condition_type):
    if condition_type == "canny":
        return CannyDetector()
    elif condition_type == "depth" or condition_type == "normal":
        return MidasDetector()
    elif condition_type == "scribble":
        return ScribbleDetector()

# load rembg
bg_remover = rembg.new_session()

detector = get_annotator(opt.condition_type)

# process function
def process(asset_name, input_image, image_need_procs, prompt, prompt_neg='', guidance_scale=7.5, input_elevation=0, input_num_steps=30, blind_control_until_step=50, input_seed=42, canny_low_threshold=None, canny_high_threshold=None):

    # seed
    kiui.seed_everything(input_seed)

    save_dir = os.path.join(opt.workspace, f"mvcontrol_{opt.condition_type}", asset_name)
    os.makedirs(save_dir, exist_ok=True)
    output_video_path = os.path.join(save_dir, "coarse_gs.mp4")
    output_ply_path = os.path.join(save_dir, "coarse_gs.ply")
    
    assert input_image is not None
    input_image = np.array(input_image)

    if image_need_procs:
        if input_image.shape[-1] == 3:
            image_rgba = rembg.remove(input_image, session=bg_remover)
        else:
            image_rgba = input_image
        image_rgba = pad_to_square(image_rgba)
        image_rgba = cv2.resize(image_rgba, (256, 256))
        mask = image_rgba[..., -1] > 127.5
        image = image_rgba.astype(np.float32) / 255
        # image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        image = (image * 255).clip(0, 255).astype(np.uint8)

        if opt.condition_type == "canny":
            input_image = detector(image, canny_low_threshold, canny_high_threshold)
        elif opt.condition_type == "depth":
            input_image, _ = detector(image, mask)
        elif opt.condition_type == "normal":
            _, input_image = detector(image, mask)
        elif opt.condition_type == "scribble":
            input_image = detector(image, 256, 256)

        # save the processed data
        Image.fromarray(image_rgba).save(
            os.path.join(save_dir, f"{asset_name}_rgba.png")
        )
        Image.fromarray(mask).save(
            os.path.join(save_dir, f"{asset_name}_mask.png")
        )
        Image.fromarray(input_image).save(
            os.path.join(save_dir, f"{asset_name}_{opt.condition_type}.png")
        )
        
    with open(os.path.join(save_dir, f"{asset_name}_prompt.txt"), "w") as f:
        f.write(prompt)

    condition = input_image.copy()
    input_image = Image.fromarray(input_image)

    c2ws = get_camera(4, elevation=0)
    
    mv_image_uint8 = pipe_mvcontrol(
        prompt, input_image, c2ws, 
        negative_prompt=prompt_neg,
        num_inference_steps=input_num_steps,
        guidance_scale=guidance_scale,
        return_dict=False,
        output_type='numpy',
        blind_control_until_step=blind_control_until_step,
    )
    mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)

    # bg removal
    mv_image = []
    for i in range(4):
        image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
        # to white bg
        image = image.astype(np.float32) / 255
        image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        mv_image.append(image)
 
    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[0], mv_image[1]], axis=1),
        np.concatenate([mv_image[2], mv_image[3]], axis=1),
    ], axis=0)
    Image.fromarray((mv_image_grid * 255).astype(np.uint8)).save(
        os.path.join(save_dir, "mv_image_grid.png")
    )

    # generate gaussians
    input_image = np.stack(mv_image, axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, output_ply_path)
        
        # render 360 video 
        images = []
        elevation = 0
        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(output_video_path, images, fps=30)

    return mv_image_grid, output_video_path, output_ply_path, condition

# gradio UI

_TITLE = '''LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation'''

_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://me.kiui.moe/lgm/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/3DTopia/LGM"><img src='https://img.shields.io/github/stars/3DTopia/LGM?style=social'/></a>
</div>

* Input can be only text, only image, or both image and text. 
* If you find the output unsatisfying, try using different seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            # whether image requires preprocess
            input_image_need_preprocess = gr.Checkbox(label="image need preprocess")
            # asset name
            asset_name = gr.Textbox(label="name")
            # input image
            input_image = gr.Image(label="image", type='pil')
            # input prompt
            input_text = gr.Textbox(label="prompt")
            # negative prompt
            input_neg_text = gr.Textbox(label="negative prompt", value='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate')
            # guidance scale
            guidance_scale = gr.Slider(label="guidance scale", minimum=1, maximum=30, step=1, value=9)
            # elevation
            input_elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)
            # inference steps
            input_num_steps = gr.Slider(label="inference steps", minimum=1, maximum=100, step=1, value=50)
            # blind control steps
            blind_control_until_step = gr.Slider(label="blind control until step", minimum=0, maximum=50, step=1, value=50)
            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=42)
            # gen button
            button_gen = gr.Button("Generate")

            if opt.condition_type == "canny":
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=75, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=175, step=1)

        
        with gr.Column(scale=1):
            with gr.Tab("Video"):
                # final video results
                output_video = gr.Video(label="video")
                # ply file
                output_file = gr.File(label="ply")
            with gr.Tab("Multi-view Image"):
                # multi-view results
                output_image = gr.Image(interactive=False, show_label=False)
            with gr.Tab("Input Condition"):
                output_condition = gr.Image(interactive=False, show_label=False)


        inputs = [
            asset_name, 
            input_image, 
            input_image_need_preprocess,
            input_text, 
            input_neg_text, 
            guidance_scale, 
            input_elevation, 
            input_num_steps, 
            blind_control_until_step,
            input_seed
        ]
        if opt.condition_type == "canny":
            inputs += [low_threshold, high_threshold]
        
        button_gen.click(
            process, 
            inputs=inputs, 
            outputs=[output_image, output_video, output_file, output_condition]
        )
    
    cond_example_paths = glob.glob(f"./load/conditions/*_{opt.condition_type}.png")
    gr.Examples(
        # examples=[
        #     "data_test/anya_rgba.png",
        #     "data_test/bird_rgba.png",
        #     "data_test/catstatue_rgba.png",
        # ],
        examples=cond_example_paths,
        inputs=[input_image],
        outputs=[output_image, output_video, output_file],
        fn=lambda x: process(input_image=x, prompt=''),
        cache_examples=False,
        label='x-to-3D Examples'
    )

    gr.Examples(
        examples=[
            "a motorbike",
            "a hamburger",
            "a furry red fox head",
        ],
        inputs=[input_text],
        outputs=[output_image, output_video, output_file],
        fn=lambda x: process(input_image=None, prompt=x),
        cache_examples=False,
        label='Text-to-3D Examples'
    )
    
block.launch(server_name="0.0.0.0", share=False)