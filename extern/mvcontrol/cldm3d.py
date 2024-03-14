import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import pytorch_lightning as pl
from einops import rearrange
import imageio.v3 as imageio
import glob

from .attention import (
    set_self_attn_processor,
    set_self_attn_trainable,
    CrossViewAttnProcessor,
    XFormersCrossViewAttnProcessor,
)
from .utils.typing import *
from .pipeline_mvcontrol import MVControlPipeline
# from .controlnet3d import ControlNet3DModel
from .mvcontrolnet import MultiViewControlNetModel
from .scheduler import DDIMScheduler_ as DDIMScheduler
from .camera_proj import CameraMatrixEmbedding
from .unet import UNet2DConditionModel

from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

import diffusers
from diffusers import (
    ModelMixin,
    AutoencoderKL,
    ControlNetModel,
    # DDIMScheduler,
    # UNet2DConditionModel,
    DiffusionPipeline,
    Transformer2DModel
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor, 
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor, 
    AttnProcessor
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionSafetyChecker, 
    StableDiffusionPipelineOutput
)
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class ControlPose3dModel(pl.LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        pretrained_controlnet2d_path: Optional[str] = None,
        pretrained_mvcontrolnet_path: Optional[str] = None,
        num_views: int = 4,
        unet_use_cross_view_attention: bool = False,
        unet_self_attention_trainable: bool = False,
        unet_cross_attention_trainable: bool = False,
        unet_whole_trainable: bool = False,
        use_controlnet: bool = False,
        condition_drop_probs: dict[str, float] = None,
        camera_matrix_embedding_dim: Optional[int] = 768,
        camera_embed_add_to: Literal["cross_attn", "resnet"] = "resnet",
        guidance_scale: float = 9.0,
        negative_prompt: str = None,
        num_inference_steps: int = 20,
        loss_type: str = 'l2',
        lr: float = 1e-5,
        lr_scheduler_type: str = 'constant_with_warmup',
        warmup_steps: int = 100,
        reso: int = 256,
        output_dir: str = './outputs',
        guess_mode: bool = False,
        sample_from_same_latent: bool = False,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'fp16',
        init_device = 'cpu',
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
        p_batch_2d: float = 0.3,
        controlnet_version: Literal["origin", "mvdream"] = "mvdream",
        **kwargs
    ):
        
        super().__init__()

        self.save_hyperparameters()
        # Load scheduler, tokenizer and models.
        logger.info(f'[INIT] Loading pretrained stable diffusion from {pretrained_model_name_or_path}...')
        noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        # Load mvcontrolnet
        if use_controlnet:
            if pretrained_mvcontrolnet_path is not None:
                controlnet = MultiViewControlNetModel.from_pretrained(pretrained_mvcontrolnet_path)
                logger.info(f"[INIT] Initialize MultiViewControlNetModel from {pretrained_mvcontrolnet_path}.")
            elif pretrained_controlnet2d_path is not None:
                logger.info(f"[INIT] Initialize MultiViewControlNetModel by copying 2D ControlNetModel {pretrained_controlnet2d_path}.")
                controlnet = ControlNetModel.from_pretrained(pretrained_controlnet2d_path)
                controlnet = MultiViewControlNetModel.from_controlnet2d(
                    controlnet,
                    num_views=num_views,
                    version=controlnet_version,
                    controlnet_block_condition_type=camera_embed_add_to,
                )
            else:
                logger.info(f"[INIT] Initialize ControlNet3DModel by copying Stable Diffusion.")
                controlnet = MultiViewControlNetModel.from_unet(
                    unet,
                    num_views=num_views,
                    version=controlnet_version,
                    controlnet_block_condition_type=camera_embed_add_to,
                )
        else:
            controlnet = None
        
        # Freeze vae, text_encoder and unet
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(unet_whole_trainable)
        
        # mixed precision
        weights_dtype = torch.float32
        if mixed_precision == "fp16":
            weights_dtype = torch.float16
        elif mixed_precision == "bf16":
            weights_dtype = torch.bfloat16
        self.weights_dtype = weights_dtype

        # Move unet, vae text_encoder and our model to device and cast to weight_dtype
        unet.to(init_device, dtype=self.weights_dtype)
        vae.to(init_device, dtype=self.weights_dtype)
        text_encoder.to(init_device, dtype=self.weights_dtype)
        if controlnet is not None:
            controlnet.to(init_device, dtype=self.weights_dtype)

        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer
        self.vae = vae
        self.text_encoder = text_encoder
        self.unet = unet
        self.controlnet = controlnet

        self.check_xformers_attn()
        
        # Maybe unet use cross view attention
        if unet_use_cross_view_attention:
            # attn_procs_cls = XFormersCrossViewAttnProcessor if self.use_xformers_attn else CrossViewAttnProcessor
            # set_self_attn_processor(unet, attn_procs_cls(num_views=num_views, batch_size=batch_size_train))
            self._set_unet_self_attn_cross_view_processor()
            logger.info('[INIT] With `unet_use_cross_view_attention` set as True, enable unet cross view attention.')
            
        if controlnet_version == "mvdream":
            self._set_controlnet_self_attn_cross_view_processor()
            logger.info('[INIT] MultiViewControlNet self attention use cross view attention processors!')


        if unet_whole_trainable:
            logger.info('[INIT] Turn the whole unet trainable!')
        else:
            # Maybe train unet's attention params
            if unet_self_attention_trainable:
                logger.info('[INIT] Set unet self attention layers trainable.')
                self._set_unet_self_attn_trainable()

            if unet_cross_attention_trainable:
                logger.info('[INIT] Set unet cross attention layers trainable.')
                self._set_unet_cross_attn_trainable()
        
        # # Create feature renderer
        # self.renderer = TriplaneFeatureRenderer(
        #     bound=bound, 
        #     num_sample_points=n_points_per_ray,
        #     fov=fov
        # )

        # Camera extrinsic matrix embedding
        if "camera_proj" in os.listdir(pretrained_model_name_or_path):
            logger.info("[INIT] Found camera embedding dir in the pretrained model path, loading pretrained camera embedding from it.")
            self.camera_proj = CameraMatrixEmbedding.from_pretrained(
                pretrained_model_name_or_path, subfolder="camera_proj"
            )
            self.camera_proj.requires_grad_(False)
        else:
            assert camera_matrix_embedding_dim is not None
            self.camera_proj = CameraMatrixEmbedding(
                in_channels=16,
                camera_embed_dim=camera_matrix_embedding_dim,
                act_fn=self.unet.config.act_fn,
                cond_add_to=camera_embed_add_to,
            )
        self.camera_proj.to(init_device, dtype=self.weights_dtype)

        # Create pipeline
        self.pipeline = MVControlPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            camera_proj=self.camera_proj,
            scheduler=self.noise_scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            device=init_device
        )

        # Since our model has two conditions: text and poses
        # we design to randomly use different conditions in training
        self.condition_drop_probs = condition_drop_probs if condition_drop_probs is not None else {
            "drop_hint": 0.1,
            "drop_text": 0.1,
            "drop_both": 0.2
        }
        # assert np.sum(self.condition_drop_probs.values()) == 1

        self.loss_type = loss_type
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.negative_prompt = negative_prompt
        self.output_dir = output_dir


    def set_save_dir(self, dir):
        self.output_dir = dir
        os.makedirs(os.path.join(self.output_dir, 'validation'), exist_ok=True)

        self.validation_out_samples = []
        self.validation_out_prompts = []

    def _make_trainable(self, module):
        module.to(dtype=torch.float32)
        module.requires_grad_(True)

    def _set_unet_self_attn_trainable(self):
        def fn_recursive_attn_trainable(module: torch.nn.Module):
            if isinstance(module, Transformer2DModel) and hasattr(module, "transformer_blocks"):
                for sub_module in module.transformer_blocks:
                    self._make_trainable(sub_module.attn1)    # self-attention trainable
            else:
                for name, sub_module in module.named_children():
                    fn_recursive_attn_trainable(sub_module)
                    
        for name, module in self.unet.named_children():
            fn_recursive_attn_trainable(module)

    def _set_unet_cross_attn_trainable(self):
        def fn_recursive_attn_trainable(module: torch.nn.Module):
            if isinstance(module, Transformer2DModel) and hasattr(module, "transformer_blocks"):
                for sub_module in module.transformer_blocks:
                    self._make_trainable(sub_module.attn2)    # self-attention trainable
            else:
                for name, sub_module in module.named_children():
                    fn_recursive_attn_trainable(sub_module)
                    
        for name, module in self.unet.named_children():
            fn_recursive_attn_trainable(module)
            
    def _set_unet_self_attn_cross_view_processor(self):
        attn_procs_cls = XFormersCrossViewAttnProcessor if self.use_xformers_attn else CrossViewAttnProcessor
        set_self_attn_processor(
            self.unet, attn_procs_cls(
                num_views=self.hparams.num_views
            )
        )

    def _set_controlnet_self_attn_cross_view_processor(self):
        attn_procs_cls = XFormersCrossViewAttnProcessor if self.use_xformers_attn else CrossViewAttnProcessor
        set_self_attn_processor(
            self.controlnet, attn_procs_cls(
                num_views=self.hparams.num_views
            )
        )
        
    def _set_unet_self_attn_vanilla_processor(self):
        attn_procs_cls = XFormersAttnProcessor if self.use_xformers_attn else AttnProcessor
        set_self_attn_processor(self.unet, attn_procs_cls())

    def _set_controlnet_self_attn_vanilla_processor(self):
        attn_procs_cls = XFormersAttnProcessor if self.use_xformers_attn else AttnProcessor
        set_self_attn_processor(self.controlnet, attn_procs_cls())
        
            
    def _get_unet_self_attn_params(self, lr=None):
        params_list = []
        
        def fn_recursive_get_self_attn_params(module: torch.nn.Module, params_list, lr=None):
            if isinstance(module, Transformer2DModel) and hasattr(module, "transformer_blocks"):
                for sub_module in module.transformer_blocks:
                    if lr is not None:
                        params_list += [{'params': sub_module.attn1.parameters(), 'lr': lr}]
                    else:
                        params_list += list(sub_module.attn1.parameters())
            else:
                for name, sub_module in module.named_children():
                    fn_recursive_get_self_attn_params(sub_module, params_list, lr)
        
        for name, module in self.unet.named_children():
            fn_recursive_get_self_attn_params(module, params_list, lr)
            
        return params_list
    
    def _get_unet_cross_attn_params(self, lr=None):
        params_list = []
        
        def fn_recursive_get_self_attn_params(module: torch.nn.Module, params_list, lr=None):
            if isinstance(module, Transformer2DModel) and hasattr(module, "transformer_blocks"):
                for sub_module in module.transformer_blocks:
                    if lr is not None:
                        params_list += [{'params': sub_module.attn2.parameters(), 'lr': lr}]
                    else:
                        params_list += list(sub_module.attn2.parameters())
            else:
                for name, sub_module in module.named_children():
                    fn_recursive_get_self_attn_params(sub_module, params_list, lr)
        
        for name, module in self.unet.named_children():
            fn_recursive_get_self_attn_params(module, params_list, lr)
            
        return params_list


    def init_lora_attn(self, model):
        lora_attn_procs_cls = LoRAXFormersAttnProcessor if self.use_xformers_attn else LoRAAttnProcessor

        lora_attn_procs = {}
        for name in model.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = model.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(model.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = model.config.block_out_channels[block_id]
            procs = lora_attn_procs_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=2)
            lora_attn_procs[name] = procs
        return lora_attn_procs

    def check_xformers_attn(self):
        """
        if xformers available, then use xformers_memory_efficient_attn
        """
        if is_xformers_available():
            logger.info('[INIT] Using xformers attention')
            self.use_xformers_attn = True
            self.unet.enable_xformers_memory_efficient_attention()
            if self.controlnet is not None:
                self.controlnet.enable_xformers_memory_efficient_attention()
        else:
            logger.info('[INIT] xformers is not available, use simple attention')
            self.use_xformers_attn = False

    @torch.no_grad()
    def get_input(self, batch):
        
        # p = np.random.rand()
        # batch_2d = p < self.hparams.p_batch_2d
        batch_2d = (batch["data_2d"] == 1).all()
        if batch_2d:
            self._set_unet_self_attn_vanilla_processor()
            if self.hparams.controlnet_version == "mvdream":
                self._set_controlnet_self_attn_vanilla_processor()
        else:
            self._set_unet_self_attn_cross_view_processor()
            if self.hparams.controlnet_version == "mvdream":
                self._set_controlnet_self_attn_cross_view_processor()
        
        # x: Float[Tensor, 'B N C H W'] = batch["images_2d"] if batch_2d else batch["images"]
        # hint = batch["hint_2d"] if batch_2d else batch["hint"]
        # c2ws = None if batch_2d else batch["c2ws"]
        # c_text = batch["prompt_2d"] if batch_2d else batch["prompt"]
        x = batch["images"]
        hint = batch["hint"]
        c2ws = None if batch_2d else batch["c2ws"]
        c_text = batch["prompt"]

        if batch_2d:
            c_ = []
            for c in c_text:
                c_ += c.split("<cldm3d_lzq>")
            c_text = c_
        
        batch_size, num_views = x.shape[:2]
        x = rearrange(x, 'b n c h w -> (b n) c h w')
        x = x.to(memory_format=torch.contiguous_format)

        # controlnet_cond: Float[Tensor, 'B C H W'] = batch['hint']

        x = self.pipeline.encode_image(
            x.to(dtype=self.pipeline.vae.dtype)
        )
        # controlnet_cond = self.pipeline.encode_image(
        #     controlnet_cond.to(dtype=self.pipeline.vae.dtype)
        # )
        if not batch_2d:
            x = rearrange(x, '(b n) c h w -> b n c h w', n=num_views)
        else:
            x = x.unsqueeze(1)
            hint = rearrange(hint, 'b n c h w -> (b n) c h w')
        
        batch_size = x.shape[0]

        ps = np.random.rand(batch_size)    # control condition dropping
        cd_probs = self.condition_drop_probs
        for i, p in enumerate(ps):
            if p < cd_probs['drop_both']:
                c_text[i] = ""
                hint[i] = torch.zeros_like(hint[i])
            elif p < cd_probs['drop_both'] + cd_probs['drop_hint']:
                hint[i] = torch.zeros_like(hint[i])
            elif p < cd_probs['drop_both'] + cd_probs['drop_hint'] + cd_probs['drop_text']:
                c_text[i] = ""
            else:
                pass    # full condition      

        c_text = self.pipeline._encode_prompt(
            c_text,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        ).to(memory_format=torch.contiguous_format)

        # c2ws = batch['c2ws']
        cond = {
            'c_text': c_text,
            'c_control': hint,
            'c_pose': c2ws,
            'is_batch_2d': batch_2d,
        }
        del batch
        
        return x, cond 
    
    
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target.float(), pred.float())
            else:
                loss = torch.nn.functional.mse_loss(target.float(), pred.float(), reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def forward(
        self, 
        x_noisy, 
        controlnet_cond, 
        t, 
        encoder_hidden_states,
        c2ws,
        forward_2d,
        **kwargs
    ):

        # compute p_loss, only support eps prediction here
        noise_pred = self.pipeline._forward(
            latent_model_input=x_noisy,
            controlnet_cond=controlnet_cond,
            t=t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_conditioning_scale=1.0,
            c2ws=c2ws,
            forward_2d=forward_2d
        )

        return noise_pred

    def training_step(self, batch):
        x, cond = self.get_input(batch)
        t = torch.randint(0, self.pipeline.scheduler.num_train_timesteps, 
            (x.shape[0],), device=self.device
        ).long()
        # add noise 
        noise = torch.randn_like(x)
        x_noisy = self.noise_scheduler.add_noise(x, noise, t)
        
        noise_pred = self(
            x_noisy, 
            cond['c_control'],
            t, 
            cond['c_text'],
            cond['c_pose'],
            forward_2d=cond['is_batch_2d']
        )
        del cond

        # here only compute the simple loss disgarding logvar learning and vlb_loss
        loss = self.get_loss(noise_pred, noise)
        
        # self.log("train/loss", loss, prog_bar=True, on_step=True, rank_zero_only=True)
        # self.log("train/lr", self.learning_rate, prog_bar=True, on_step=True, rank_zero_only=True)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        prompts = batch['prompt']
        c2ws = batch['c2ws']
        sample = self.pipeline(
            prompt=prompts,
            image=None,
            c2ws=c2ws,
            height=self.hparams.reso,
            width=self.hparams.reso,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=[self.negative_prompt]*len(prompts),
            num_images_per_prompt=1,
            output_type='decode',
            return_dict=False,
            guess_mode=self.hparams.guess_mode,
            start_from_same_latent=self.hparams.sample_from_same_latent
        )   # (-1, 1)

        self.validation_out_samples.append(sample)
        self.validation_out_prompts.append(batch['prompt'])



    def on_validation_epoch_end(self) -> None:
        
        sample = torch.cat(self.validation_out_samples, dim=0)
        prompts = []
        for p in self.validation_out_prompts:
            prompts += p

        # post process
        sample = (sample * 127.5 + 127.5).cpu()
        batch_size, num_views = sample.shape[:2]
        sample = rearrange(sample, 'b n c h w -> (b n) c h w')
        grid = make_grid(sample, nrow=num_views)
        grid = rearrange(grid, 'c h w -> h w c')
        grid = grid.numpy().clip(0, 255).astype(np.uint8)

        # save
        val_out_dir = os.path.join(
            self.output_dir, 'validation', f'epoch{self.current_epoch}-it{self.global_step}'
        )
        os.makedirs(val_out_dir, exist_ok=True)
        imageio.imwrite(
            os.path.join(val_out_dir, 'samples.png'), 
            grid
        )
        with open(os.path.join(val_out_dir, 'prompts.json'), 'w') as f:
            json.dump(prompts, f, indent=4)

        # clear
        self.validation_out_samples = []
        self.validation_out_prompts = []

    def configure_optimizers(self):
        lr = self.hparams.lr
        params = []
        if self.controlnet is not None:
            params += list(self.controlnet.parameters())

        if self.hparams.unet_whole_trainable:
            params += list(self.unet.parameters())
        else:
            if self.hparams.unet_self_attention_trainable:
                params += self._get_unet_self_attn_params()
            if self.hparams.unet_cross_attention_trainable:
                params += self._get_unet_cross_attn_params()

        # if self.camera_proj is not None and self.camera_proj.requires_grad:
        #     params += list(self.camera_proj.parameters())

        optimizer = torch.optim.AdamW(
            params, 
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-3,
            eps=1e-8
        )

        lr_scheduler = get_scheduler(
            self.hparams.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }






    






