import os
import sys
import bisect
from dataclasses import dataclass, field

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# from extern.cldm3d.utils.camera import convert_blender_to_opencv, normalize_camera

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from threestudio.utils.config import config_to_primitive

from extern.mvcontrol.unet import UNet2DConditionModel
from extern.mvcontrol.mvcontrolnet import MultiViewControlNetModel
from extern.mvcontrol.camera_proj import CameraMatrixEmbedding
from extern.mvcontrol.pipeline_mvcontrol import MVControlPipeline
from extern.mvcontrol.attention import (
    CrossViewAttnProcessor, XFormersCrossViewAttnProcessor,
    set_self_attn_processor
)
from extern.mvcontrol.scheduler import DDIMScheduler_ as DDIMScheduler
from extern.mvcontrol.utils.camera import normalize_camera, get_camera

# from diffusers import DDIMScheduler
from diffusers.utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

@threestudio.register("mvcontrol-guidance")
class MultiviewControlNetGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "local_path/pretrained"
        pretrained_controlnet_name_or_path: Optional[str] = None
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256

        weighting_strategy: str = "sds"
        score_distillation_type: str = "sds"    # [sds, sds_recon, nfsd, csd]
        recon_std_rescale: float = 0.5
        nfsd_milestone: int = 200
        csd_text_weight: float = 1.
        csd_neg_weight: float = 0.5
        csd_neg_weight_milestones: List[int] = field(default_factory=lambda: [])
        

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview ControlNet...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "trust_remote_code": True,
        }

        # Load MVControl model
        if self.cfg.pretrained_controlnet_name_or_path is not None:
            controlnet = MultiViewControlNetModel.from_pretrained(
                self.cfg.pretrained_controlnet_name_or_path, torch_dtype=self.weights_dtype
            )
        elif "controlnet" in os.listdir(self.cfg.pretrained_model_name_or_path):
            controlnet = MultiViewControlNetModel.from_pretrained(
                self.cfg.pretrained_model_name_or_path, subfolder="controlnet", torch_dtype=self.weights_dtype
            )
        else:
            raise ValueError("You must provide `pretrained_controlnet_name_or_path` or there must exists a folder named `controlnet` under `pretrained_model_name_or_path`.")

        # Load pipeline
        self.pipe = MVControlPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,     # mvdream
            controlnet=controlnet,
            **pipe_kwargs
        )

        # Enable cross-view attention for the unet
        if is_xformers_available():
            import xformers
            attn_procs_cls = XFormersCrossViewAttnProcessor
        else:
            attn_procs_cls = CrossViewAttnProcessor
        set_self_attn_processor(
            self.pipe.unet, attn_procs_cls(num_views=self.cfg.n_view)
        )
        if self.pipe.controlnet is not None:
            set_self_attn_processor(
                self.pipe.controlnet, attn_procs_cls(num_views=self.cfg.n_view)
            )

        # if self.cfg.enable_sequential_cpu_offload:
        #     self.pipe.enable_sequential_cpu_offload()

        # if self.cfg.enable_attention_slicing:
        #     self.pipe.enable_attention_slicing(1)

        # if self.cfg.enable_channels_last_format:
        #     self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.camera_proj = self.pipe.camera_proj.eval()
        self.controlnet = self.pipe.controlnet.eval()
        self.set_modules_require_grad_false(
            [self.vae, self.unet, self.camera_proj, self.controlnet]
        )

        self.scheduler = self.pipe.scheduler
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.set_min_max_steps(min_step_percent, max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)

        # Register 4 relative canonical camera matrices
        self.c2ws_relative = get_camera(
            num_frames=self.cfg.n_view, elevation=0
        ).reshape(-1, 4, 4).to(self.device, dtype=self.weights_dtype)

        self.score_distillation_type = self.cfg.score_distillation_type
        self.csd_text_weight = self.cfg.csd_text_weight
        self.csd_neg_weights = (
            [self.cfg.csd_neg_weight] 
            if isinstance(self.cfg.csd_neg_weight, float) 
            else config_to_primitive(self.cfg.csd_neg_weight)
        )
        self.csd_neg_weight = self.csd_neg_weights[0]
        if len(self.csd_neg_weights) == 1:
            if len(self.cfg.csd_neg_weight_milestones) > 0:
                threestudio.warn(
                    "Ignoring csd_neg_weight_milestones since csd_neg_weight is not changing"
                )
            self.csd_neg_weight_milestones = [-1]
        else:
            assert len(self.csd_neg_weights) == len(self.cfg.csd_neg_weight_milestones) + 1
            self.csd_neg_weight_milestones = [-1] + self.cfg.csd_neg_weight_milestones

        threestudio.info(f"Loaded Multiview ControlNet!")

    def set_modules_require_grad_false(self, modules: List[torch.nn.Module]):
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def get_camera_cond(self, camera: Float[Tensor, "B 4 4"]):
        # Note: the input of threestudio is blender coordinate system
        # But our model is trained on opencv coordinate system
        # camera = convert_blender_to_opencv(camera)
        camera = normalize_camera(camera)
        return camera

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)
    
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        c2w: Float[Tensor, "B 4 4"],
        controlnet_condition: Float[Tensor, "H W C"] = None,
        rgb_as_latents: bool = False,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        drop_control: bool = False,
        guidance_scale: float = None,
        **kwargs,
    ):  
        n_view = self.cfg.n_view
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        batch_size = rgb.shape[0] // n_view
        camera = c2w if drop_control else self.c2ws_relative

        # Prepare conditions
        if text_embeddings is None:
            embedding_list = [
                prompt_utils.uncond_text_embeddings, 
                prompt_utils.text_embeddings, 
            ]
            if self.score_distillation_type in ["nfsd", "csd"]:
                embedding_list += [prompt_utils.null_text_embeddings]

            text_embeddings = torch.cat(embedding_list, dim=0).repeat_interleave(batch_size, dim=0)   # unconditional first

        if controlnet_condition is not None:
            controlnet_condition = controlnet_condition.permute(2, 0, 1)
            controlnet_condition = torch.stack(
                [controlnet_condition]*len(embedding_list), dim=0
            ).repeat_interleave(batch_size, dim=0)
            # controlnet_condition = torch.stack(
            #     [torch.zeros_like(controlnet_condition), controlnet_condition], dim=0
            # ).repeat_interleave(batch_size, dim=0)

        if camera is not None:
            camera = self.get_camera_cond(camera)
            camera = rearrange(camera, "(b n) i j -> b n (i j)", n=n_view)
            camera = torch.cat([camera] * len(embedding_list), dim=0).to(text_embeddings)

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(rgb_BCHW, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(rgb_BCHW, (self.cfg.image_size, self.cfg.image_size), mode='bilinear', align_corners=False)
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)
        latents = rearrange(latents, "(b n) c h w -> b n c h w", n=n_view)

        # sample timestep
        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        # t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        if not guidance_scale:
            guidance_scale = self.cfg.guidance_scale
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)   # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * len(embedding_list), dim=0)
            noise_pred = self.pipe._forward(
                latent_model_input, 
                controlnet_condition,
                t[0],
                encoder_hidden_states=text_embeddings.to(self.unet.dtype),
                c2ws=camera.to(self.camera_proj.dtype),
            )
        if self.score_distillation_type in ["sds", "sds_recon"]:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_uncond, noise_pred_text, noise_pred_null = noise_pred.chunk(3)
            if self.score_distillation_type == "nfsd":
                if t[0] > self.cfg.nfsd_milestone:
                    noise_pred = noise_pred_null - noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_null + guidance_scale * (noise_pred_text - noise_pred_uncond)
            elif self.score_distillation_type == "csd":
                noise_pred = (
                    self.csd_text_weight * noise_pred_text
                    + (self.csd_neg_weight - self.csd_text_weight) * noise_pred_null
                    - self.csd_neg_weight * noise_pred_uncond
                )

        noise_pred = rearrange(noise_pred, "b n c h w -> (b n) c h w")
        latents_noisy = rearrange(latents_noisy, "b n c h w -> (b n) c h w")
        latents = rearrange(latents, "b n c h w -> (b n) c h w")

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "ism":
            w = (((1 - self.alphas[t]) / self.alphas[t]) ** 0.5).reshape(-1, 1, 1, 1)

        if self.score_distillation_type == "sds":
            grad = w * (noise_pred - noise)

            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        elif self.score_distillation_type == "sds_recon":
            # reconstruct x0
            self.scheduler: DDIMScheduler
            latents_recon = pred_x0(self.scheduler, latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = pred_x0(self.scheduler, latents_noisy, t, noise_pred)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.cfg.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.cfg.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = self.cfg.recon_std_rescale * latents_recon_adjust + (1-self.cfg.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        elif self.score_distillation_type in ["nfsd", "csd"]:
            grad = w * noise_pred
            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        else:
            raise ValueError(f"Unimplemented score distillation type: {self.score_distillation_type}!")


        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.set_min_max_steps(min_step_percent, max_step_percent)

        if self.score_distillation_type == "csd":
            weight_ind = bisect.bisect_right(self.csd_neg_weight_milestones, global_step) - 1
            self.csd_neg_weight = self.csd_neg_weights[weight_ind]



def pred_x0(
    self,
    sample: torch.FloatTensor,
    timesteps: torch.IntTensor,
    model_output: torch.FloatTensor,
) -> torch.FloatTensor:
    # 1. device
    device = timesteps.device

    # 2. compute alphas, betas
    alphas_cumprod = self.alphas_cumprod.to(device)
    alpha_prod_t = alphas_cumprod[timesteps]

    beta_prod_t = 1 - alpha_prod_t

    if timesteps.ndim > 0:
        beta_prod_t = beta_prod_t[:, None, None, None]
        alpha_prod_t = alpha_prod_t[:, None, None, None]

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )

    # # 4. Clip or threshold "predicted x_0"
    # if self.config.thresholding:
    #     pred_original_sample = self._threshold_sample(pred_original_sample)
    # elif self.config.clip_sample:
    #     pred_original_sample = pred_original_sample.clamp(
    #         -self.config.clip_sample_range, self.config.clip_sample_range
    #     )
    return pred_original_sample