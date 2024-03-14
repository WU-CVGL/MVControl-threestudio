import os
import math
import bisect
import random
from easydict import EasyDict
from dataclasses import dataclass, field
from PIL import Image

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.utils.base import BaseModule
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *
from threestudio.utils.ops import (
    binary_cross_entropy, 
    dot, 
    get_projection_matrix, 
    get_mvp_matrix, 
    get_ray_directions,
    get_rays
)

from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing
import open3d as o3d

from .base import BaseSuGaRSystem
from ..geometry.gaussian_base import BasicPointCloud, Camera
from ..geometry.sugar import SuGaRModel
from ..utils.sugar_utils import SuGaRRegularizer

from extern.mvcontrol.utils.image import HWC3, resize_image

from torchvision.utils import save_image

class ReferenceCamera(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        num_views: int = 4
        camera_distance: float = 1.0
        camera_distance_learnable: bool = False
        init_elevation_deg: float = 0.
        elevation_learnable: bool = False
        init_azimuth_deg: float = 0.
        azimuth_learnable: bool = False
        relative_radius: bool = False
        fovy_deg: float = 49.13
        height: Any = 64
        width: Any = 64
        resolution_milestones: Any = None

    cfg: Config

    def configure(self) -> None:
        self.num_views = self.cfg.num_views
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)

        if len(self.heights) == 1:
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]

        self.fovy = self.cfg.fovy_deg * torch.pi / 180
        self.focal_length = 0.5 * self.height / np.tan(0.5 * self.fovy)

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.directions_unit_focal = self.directions_unit_focals[0]

        self.elevation = nn.Parameter(
            torch.as_tensor(
                self.cfg.init_elevation_deg / 180 * math.pi, 
                device=self.device, dtype=torch.float32
            ),
            requires_grad=self.cfg.elevation_learnable
        )
        self.azimuth = nn.Parameter(
            torch.as_tensor(
                self.cfg.init_azimuth_deg / 180 * math.pi,
                device=self.device, dtype=torch.float32
            ),
            requires_grad=self.cfg.azimuth_learnable
        )

        camera_distance = self.cfg.camera_distance
        if self.cfg.relative_radius:
            scale = 1 / np.tan(0.5 * self.fovy)
            camera_distance *= scale
        self.camera_distance = nn.Parameter(
            torch.as_tensor(camera_distance, device=self.device, dtype=torch.float32),
            requires_grad=self.cfg.camera_distance_learnable
        )


    @property
    def position(self):
        pos: Float[Tensor, '3'] = torch.zeros(3, dtype=self.camera_distance.dtype, device=self.camera_distance.device)
        pos[0] = self.camera_distance * torch.cos(self.elevation) * torch.cos(self.azimuth)
        pos[1] = self.camera_distance * torch.cos(self.elevation) * torch.sin(self.azimuth)
        pos[2] = self.camera_distance * torch.sin(self.elevation)
        return pos
    
    @property
    def positions_multiview(self):
        positions: Float[Tensor, "Nv 3"] = torch.zeros((self.num_views, 3), dtype=self.camera_distance.dtype, device=self.device)
        interv = 2 * torch.pi / self.num_views
        for i in range(self.num_views):
            azi = self.azimuth + i * interv
            positions[i, 0] = self.camera_distance * torch.cos(self.elevation) * torch.cos(azi)
            positions[i, 1] = self.camera_distance * torch.cos(self.elevation) * torch.sin(azi)
            positions[i, 2] = self.camera_distance * torch.sin(self.elevation)
        return positions

    def camera_matrix_c2w(self, positions: Float[Tensor, "N 3"]):
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        center: Float[Tensor, 'N 3'] = torch.zeros_like(positions)
        up = torch.as_tensor([[0, 0, 1]], dtype=center.dtype, device=center.device)
        lookat = F.normalize(center - positions)
        right = F.normalize(torch.cross(lookat, up))
        up = F.normalize(torch.cross(right, lookat))
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), positions.unsqueeze(-1)],
            dim=-1
        )
        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0
        return c2w
    
    def mvp_matrix(self, positions: Float[Tensor, "N 3"]):
        N = positions.shape[0]
        proj_mtx: Float[Tensor, 'N 4 4'] = get_projection_matrix(
            torch.as_tensor([self.fovy] * N), 
            self.width / self.height,
            0.1, 
            1000.0
        ).to(self.device)
        mvp_mtx: Float[Tensor, 'N 4 4'] = get_mvp_matrix(
            self.camera_matrix_c2w(positions), proj_mtx
        )
        return mvp_mtx
    
    def get_rays(self, camera_positions, n_views, noise_scale: float = 0.0):
        directions = self.directions_unit_focal.unsqueeze(0).repeat(n_views, 1, 1, 1)
        directions[..., :2] = directions[..., :2] / self.focal_length
        rays_o, rays_d = get_rays(
            directions.to(self.device), 
            self.camera_matrix_c2w(camera_positions), 
            keepdim=True,
            noise_scale=noise_scale
        )
        return {'rays_o': rays_o, 'rays_d': rays_d}
    
    def get_batch(self, noise_scale: float = 0.0, first_view_only: bool = False):
        n_views = 1 if first_view_only else self.num_views
        batch_elevations = self.elevation.expand(n_views) / torch.pi * 180
        batch_azimuths = torch.zeros(n_views)
        for i in range(n_views):
            batch_azimuths[i] = self.azimuth + i * (2 * torch.pi / n_views)
        batch_azimuths = batch_azimuths / torch.pi * 180
        batch_distances = self.camera_distance.expand(n_views)
        
        positions_multiview = self.positions_multiview[:n_views]
        rays = self.get_rays(positions_multiview, n_views, noise_scale=noise_scale)
        return {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'mvp_mtx': self.mvp_matrix(positions_multiview),
            'camera_positions': positions_multiview,
            'c2w': self.camera_matrix_c2w(positions_multiview),
            'elevation': batch_elevations,
            'azimuth': batch_azimuths,
            'camera_distances': batch_distances,
            'height': self.height,
            'width': self.width,
            'fovy': [self.fovy] * n_views
        }
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        super().update_step(epoch, global_step, on_load_weights)
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = 0.5 * self.height / np.tan(0.5 * self.fovy)


@threestudio.register("sugar-mvcontrol-system")
class SuGaRMVControlSystem(BaseSuGaRSystem):
    @dataclass
    class Config(BaseSuGaRSystem.Config):
        stage: str = "gaussian"
        freq: dict = field(default_factory=dict)
        refinement: bool = False
        ambient_ratio_min: float = 0.5
        back_ground_color: Tuple[float, float, float] = (1, 1, 1)
        
        # =================== MVControl configs ================== #
        n_view: int = 4
        hint_image_path: Optional[str] = None
        hint_mask_path: Optional[str] = None
        hint_rgb_path: Optional[str] = None
        control_condition_type: str = "depth"

        ref_camera: dict = field(default_factory=dict)
        ref_camera_ray_noise_scale: Any = 0.0

        # MVControl
        prompt_processor_control_type: str = ""
        prompt_processor_control: dict = field(default_factory=dict)

        guidance_control_type: Optional[str] = None
        guidance_control: dict = field(default_factory=dict)

        # DeepFloyd-IF
        prompt_processor_geometry_type: str = ""
        prompt_processor_geometry: dict = field(default_factory=dict)

        guidance_geometry_type: Optional[str] = None
        guidance_geometry: dict = field(default_factory=dict)

        # ============= SuGaR regularization configs ============= #
        use_sugar_reg: bool = True
        knn_to_track: int = 16
        n_samples_for_sugar_sdf_reg: int = 500000
        # min_opac_prune: Any = 0.5


    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.stage = self.cfg.stage
        if self.stage == "gaussian":
            self.automatic_optimization = False
        # self.automatic_optimization = False
        
        # Configure reference camera
        ref_camera_cfg = self.cfg.ref_camera
        ref_camera_cfg.update({"num_views": self.cfg.n_view})
        self.ref_camera = ReferenceCamera(ref_camera_cfg)
        
        # 2D diffusion guidance
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        # self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        # mvcontrol diffusion guidance
        if self.cfg.guidance_control_type is not None:
            self.prompt_processor_control = threestudio.find(self.cfg.prompt_processor_control_type)(
                self.cfg.prompt_processor_control
            )
            self.prompt_utils_control = self.prompt_processor_control()
            self.guidance_control = threestudio.find(self.cfg.guidance_control_type)(
                self.cfg.guidance_control
            )
            self.guidance_control.requires_grad_(False)
        else:
            self.prompt_processor_control = None
            self.guidance_control = None

        # mvcontrol diffusion guidance
        if self.cfg.guidance_geometry_type is not None:
            self.prompt_processor_geometry = threestudio.find(self.cfg.prompt_processor_geometry_type)(
                self.cfg.prompt_processor_geometry
            )
            self.prompt_utils_geometry = self.prompt_processor_geometry()
            self.guidance_geometry = threestudio.find(self.cfg.guidance_geometry_type)(
                self.cfg.guidance_geometry
            )
            # self.guidance_geometry.requires_grad_(False)
        else:
            self.prompt_processor_geometry = None
            self.guidance_geometry = None
        
        # Load hint images
        # load hint image if exists
        if self.guidance_control is not None and self.cfg.hint_image_path is not None:
            threestudio.info(f"Loading hint image {self.cfg.hint_image_path}...")
            hint_image = Image.open(self.cfg.hint_image_path)
            hint_image = np.array(hint_image)
            hint_image = HWC3(hint_image)
            hint_image = resize_image(hint_image, self.guidance_control.cfg.image_size)
            hint_image = torch.from_numpy(hint_image).float() / 255.0
            self.hint_image = hint_image.to(self.geometry.device)
        else:
            self.hint_image = None

        if self.cfg.hint_mask_path is not None:
            mask = Image.open(self.cfg.hint_mask_path).convert("RGB")
            mask = np.array(mask)[..., 0] / 255
            mask = torch.as_tensor(mask.astype(bool))
            self.hint_mask = mask.to(self.geometry.device).reshape(1, *mask.shape, 1)
        else:
            self.hint_mask = None

        if self.cfg.hint_rgb_path is not None:
            rgb = Image.open(self.cfg.hint_rgb_path).convert("RGB")
            rgb = np.array(rgb)
            rgb = torch.from_numpy(rgb).float() / 255.0
            self.hint_rgb = rgb.to(self.geometry.device).unsqueeze(0)
        else:
            self.hint_rgb = None

        if self.cfg.control_condition_type == "depth" and self.hint_image is not None:
            self.ref_depth = self.hint_image[None, ..., :1].clone()
        else:
            self.ref_depth = None
            
        if self.cfg.control_condition_type == "normal" and self.hint_image is not None:
            self.ref_normal = self.hint_image[None].clone()
        else:
            self.ref_normal = None

    def configure_optimizers(self):
        optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [optim]
        if hasattr(self.cfg.optimizer, "name"):
            net_optim = parse_optimizer(self.cfg.optimizer, self)
            optim = self.geometry.merge_optimizer(net_optim)
            self.merged_optimizer = True
        else:
            self.merged_optimizer = False
        return [optim]

    def on_load_checkpoint(self, checkpoint):
        if self.stage == "gaussian":
            num_pts = checkpoint["state_dict"]["geometry._xyz"].shape[0]
            pcd = BasicPointCloud(
                points=np.zeros((num_pts, 3)),
                colors=np.zeros((num_pts, 3)),
                normals=np.zeros((num_pts, 3)),
            )
            self.geometry.create_from_pcd(pcd, 10)
            self.geometry.training_setup()
            return
        # else:
        #     self.geometry.update_texture_features(self.geometry.cfg.square_size_in_texture)

    def forward(
        self, 
        batch: Dict[str, Any], 
        compute_color_in_rasterizer: bool = False
    ) -> Dict[str, Any]:
        self.geometry.update_learning_rate(self.global_step)
        if self.stage == "sugar" and not compute_color_in_rasterizer:
            batch.update(
                {"override_color": self.geometry.get_points_rgb()}
            )
        outputs = self.renderer.batch_forward(batch)
        return outputs

    def on_fit_start(self) -> None:
        super().on_fit_start()
        
        self._render_type = "rgb"

        self.sugar_reg = None

        self.pearson = PearsonCorrCoef().to(self.device)
        
        # self.grad_scaler = torch.cuda.amp.GradScaler()

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        super().on_train_batch_start(batch, batch_idx, unused)
        if self.stage == "gaussain" and self.cfg.use_sugar_reg and self.global_step >= self.cfg.freq.start_sugar_reg:
            self.sugar_reg = SuGaRRegularizer(
                self.geometry, keep_track_of_knn=True, knn_to_track=self.cfg.knn_to_track
            )
            self.sugar_reg.reset_neighbors(self.cfg.knn_to_track)

        if self.sugar_reg is not None:
            if (
                self.global_step % self.cfg.freq.reset_neighbors == 0
                or self.geometry.pruned_or_densified
            ):
                self.sugar_reg.reset_neighbors(self.cfg.knn_to_track)

            # self.geometry.min_opac_prune = self.C(self.cfg.min_opac_prune)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "zero123"
        """
        if guidance == "ref":
            ambient_ratio = 1.0
            shading = "diffuse"
            batch_ref = self.ref_camera.get_batch(
                noise_scale=self.C(self.cfg.ref_camera_ray_noise_scale), 
                first_view_only=True
            )
            batch_ref.update(
                {
                    "light_positions": batch["light_positions"][:1].expand_as(batch_ref["camera_positions"]), 
                }
            )
            batch = batch_ref
            gt_mask = self.hint_mask
            batch["shading"] = shading
        else:
            ambient_ratio = (
                self.cfg.ambient_ratio_min
                + (1 - self.cfg.ambient_ratio_min) * random.random()
            )
            if guidance == "mvcontrol":
                batch_control = self.ref_camera.get_batch(
                    noise_scale=self.C(self.cfg.ref_camera_ray_noise_scale)
                )
                batch_control.update(
                    {
                        "light_positions": batch["light_positions"][:1].expand_as(batch_control["camera_positions"]), 
                        "override_bg_color": torch.ones(
                            [1, 3], dtype=torch.float32, device=self.device
                        ) * np.random.rand()
                    }   # random scaled background
                )
                batch = batch_control
            elif guidance == "2d":
                batch = batch

        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)
        if self._render_type == "normal" and guidance == "2d":
            guidance_inp = out["comp_normal"]
        else:
            guidance_inp = out["comp_rgb"]

        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        if guidance == "ref":
            assert self.hint_mask is not None
            gt_mask = self.hint_mask.float()    # (1, H, W, 1)
            if gt_mask.shape[1] != batch["height"]:
                gt_mask = F.interpolate(
                    gt_mask.permute(0, 3, 1, 2),
                    size=(batch["height"], batch["width"]),
                    mode="nearest-exact"
                ).permute(0, 2, 3, 1)
            gt_mask = gt_mask > 0.5

            # mask loss
            if self.hint_mask is not None and self.C(self.cfg.loss.lambda_mask) > 0:
                out_mask = out["comp_mask"]
                set_loss("mask", F.mse_loss(gt_mask.float(), out_mask))

            # depth loss
            if self.ref_depth is not None and self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["comp_depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.ref_depth is not None and self.C(self.cfg.loss.lambda_depth_rel) > 0:
                gt_depth = self.ref_depth
                if gt_depth.shape[1] != batch["height"]:
                    gt_depth = F.interpolate(
                        gt_depth.permute(0, 3, 1, 2),
                        size=(batch["height"], batch["width"]),
                        mode="bicubic"
                    ).permute(0, 2, 3, 1)
                gt_depth = gt_depth[gt_mask]
                pred_depth = out["comp_depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(pred_depth, gt_depth)
                )

            # FIXME: Can't be used for now
            # normal loss
            if self.ref_normal is not None and self.C(self.cfg.loss.lambda_normal) > 0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "Normal is required for reference normal loss, no normal is found in the output."
                    )
                gt_normal = self.ref_normal
                if gt_normal.shape[1] != batch["height"]:
                    gt_normal = F.interpolate(
                        gt_normal.permute(0, 3, 1, 2),
                        size=(batch["height"], batch["width"]),
                        mode="bicubic"
                    ).permute(0, 2, 3, 1)

                # gt_normal = 1 - 2 * gt_normal
                gt_normal = 2 * gt_normal - 1
                gt_normal = gt_normal[gt_mask.squeeze(-1)]
                    
                gb_normal = out["comp_normal"] * 2 - 1
                w2c = batch["c2w"][:, :3, :3].inverse()
                normal_viewspace = torch.einsum("bij, bhwj -> bhwi", w2c, gb_normal)
                pred_normal = F.normalize(normal_viewspace, dim=-1)

                # pred_normal = pred_normal * 0.5 + 0.5
                # pred_normal[..., 0] = 1 - pred_normal[..., 0]
                # pred_normal = pred_normal * 2 - 1
                
                # pred_normal[..., 0] = - pred_normal[..., 0]
                pred_normal = pred_normal[gt_mask.squeeze(-1)]

                set_loss(
                    "normal",
                    1 - F.cosine_similarity(pred_normal, gt_normal).mean(),
                )
                
        elif guidance == "mvcontrol":
            guidance_out_mvcontrol = self.guidance_control(
                guidance_inp, 
                self.prompt_utils_control,
                controlnet_condition=self.hint_image,
                **batch,
            )
            set_loss("sds_control", guidance_out_mvcontrol["loss_sds"])
            
        elif guidance == "2d":
            # if self.stage == "sugar":
            #     opacity_mask = out["comp_mask"]
            #     grow_mask = F.max_pool2d(
            #         1 - opacity_mask.float().permute(0, 3, 1, 2), (11, 11), 1, 5
            #     )
            #     grow_mask = (grow_mask.permute(0, 2, 3, 1) > 0.5).repeat(1, 1, 1, 3)
            #     guidance_inp[grow_mask] = guidance_inp[grow_mask].detach()
            guidance_out_2d = self.guidance(
                guidance_inp, 
                self.prompt_utils,
                **batch,
                # mask=out["comp_mask"] if "comp_mask" in out else None,
            )
            for name, value in guidance_out_2d.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    # set_loss(name.split("_")[-1], value)
                    set_loss(name.replace("loss_", ""), value)

            if self.guidance_geometry is not None:
                guidance_out_geometry = self.guidance_geometry(
                    out["comp_normal"] if self.global_step % 2 == 0 else out["comp_rgb"],
                    self.prompt_utils_geometry,
                    **batch
                )
                set_loss("sds_geometry", guidance_out_geometry["loss_sds"])

            # set_loss("sds_2d", guidance_out_2d["loss_sds"])
            # if "loss_sds_img" in guidance_out_2d:
            #     set_loss("sds_img_2d", guidance_out_2d["loss_sds_img"])

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        
            
        if self.stage == "gaussian" and self.sugar_reg is not None:
            ## cross entropy loss for opacity to make it binary
            if self.C(self.cfg.loss.lambda_opacity_binary) > 0:
                # only use in static stage
                visibility_filter = out["visibility_filter"]
                opacity = self.geometry.get_opacity.unsqueeze(0).repeat(len(visibility_filter), 1, 1)
                vis_opacities = opacity[torch.stack(visibility_filter)]
                set_loss(
                    "opacity_binary",
                    -(vis_opacities * torch.log(vis_opacities + 1e-10)
                    + (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)).mean()
                )
                
            if self.C(self.cfg.loss.lambda_sugar_density_reg) > 0:
                use_sdf_normal_reg = self.C(self.cfg.loss.lambda_sugar_sdf_normal_reg) > 0
                coarse_args = EasyDict(
                    {
                        # "outputs": out,
                        "n_samples_for_sdf_regularization": self.cfg.n_samples_for_sugar_sdf_reg,
                        "use_sdf_better_normal_loss": use_sdf_normal_reg,
                    }
                )
                dloss = self.sugar_reg.coarse_density_regulation(coarse_args)
                set_loss("sugar_density_reg", dloss["density_regulation"])
                if use_sdf_normal_reg:
                    set_loss("sugar_sdf_normal_reg", dloss["normal_regulation"])
                

        if self.stage == "sugar":
            surface_mesh = self.geometry.surface_mesh
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss(
                    "normal_consistency",
                    mesh_normal_consistency(surface_mesh)
                )
            if self.C(self.cfg.loss.lambda_laplacian_smoothing) > 0:
                set_loss(
                    "laplacian_smoothing",
                    mesh_laplacian_smoothing(surface_mesh, "uniform")
                )
                
            if self.C(self.cfg.loss.lambda_opacity_max) > 0:
                set_loss(
                    "opacity_max",
                    (self.geometry.get_opacity - 1).abs().mean()
                )

        if self.cfg.loss["lambda_rgb_tv"] > 0.0:
            loss_rgb_tv = tv_loss(out["comp_rgb"].permute(0, 3, 1, 2))
            set_loss("rgb_tv", loss_rgb_tv)

        if (
            out.__contains__("comp_depth")
            and self.cfg.loss["lambda_depth_tv"] > 0.0
        ):
            loss_depth_tv = tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
            set_loss("depth_tv", loss_depth_tv)

        if (
            out.__contains__("comp_normal")
            and self.cfg.loss["lambda_normal_tv"] > 0.0
        ):
            loss_normal_tv = tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
            set_loss("normal_tv", loss_normal_tv)

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        out.update({"loss": loss})
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        total_loss = 0.0

        if self.stage == "gaussian":
            self.log(
                "gauss_num",
                int(self.geometry.get_xyz.shape[0]),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        
        if self.global_step % self.cfg.freq.render_normal == 0:
            self._render_type == "normal"
        else:
            self._render_type == "rgb"
        
        out_2d = self.training_substep(batch, batch_idx, guidance="2d")
        total_loss += out_2d["loss"]
        
        if self.guidance_control is not None and self.C(self.cfg.loss.lambda_sds_control) > 0:
            out_mvcontrol = self.training_substep(batch, batch_idx, guidance="mvcontrol")
            total_loss += out_mvcontrol["loss"]
        
        out_ref = self.training_substep(batch, batch_idx, guidance="ref")
        total_loss += out_ref["loss"]


        self.log("train/loss", total_loss, prog_bar=True)


        if self.stage == "gaussian":
            total_loss.backward()
            
            visibility_filter = out_2d["visibility_filter"]
            radii = out_2d["radii"]
            viewspace_point_tensor = out_2d["viewspace_points"]

            self.geometry.update_states(
                self.global_step,
                visibility_filter,
                radii,
                viewspace_point_tensor,
            )

            opt.step()
            opt.zero_grad(set_to_none=True)

        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_depth"][0, :, :, 0],
                        "kwargs": {"cmap": "jet", "data_range": (0, 1)},
                    }
                ]
                if "comp_depth" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["comp_mask"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    }
                ]
                if "comp_mask" in out
                else []
            ),
            # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        # filestem = f"it{self.true_global_step}-val"
        # self.save_img_sequence(
        #     filestem,
        #     filestem,
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=30,
        #     name="validation_epoch_end",
        #     step=self.true_global_step,
        # )
    
        # Compute quantile of gaussian opacities
        n_quantiles = 10
        for i in range(n_quantiles):
            quant = self.geometry.get_opacity.quantile(i/n_quantiles).item()
            threestudio.info(f'Quantile {i/n_quantiles}: {quant:.04f}')

    def test_step(self, batch, batch_idx):
        batch.update(
            {
                "override_bg_color": torch.ones([1, 3], dtype=torch.float32, device=self.device)
            }
        )
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

        # Save the current mesh as .ply file
        if self.stage == "gaussian":
            pc_save_path = os.path.join(
                self.get_save_dir(), f"exported_gs_step{self.global_step}.ply"
            )
            self.geometry.save_ply(pc_save_path)
        else:
            self.export_mesh_to_ply()

        
