import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple
from collections import defaultdict

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from threestudio.models.geometry.base import BaseGeometry
from threestudio.utils.misc import C
from threestudio.utils.typing import *

import open3d as o3d
from pytorch3d.ops import knn_points
from pytorch3d.transforms import quaternion_apply, quaternion_invert, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV, TexturesVertex

from .gaussian_base import SH2RGB, RGB2SH
from ..utils.sugar_utils import get_one_ring_neighbors

def inverse_sigmoid(x):
    return torch.log(x/(1-x))


"""For now, the class only contains functions for IO and refinement"""
@threestudio.register("sugar")
class SuGaRModel(BaseGeometry):
    @dataclass
    class Config(BaseGeometry.Config):
        sh_levels: int = 1
        position_lr: Any = 0.001
        feature_lr: Any = 0.01
        opacity_lr: Any = 0.05
        scaling_lr: Any = 0.005
        rotation_lr: Any = 0.005

        learnable_positions: bool = False
        triangle_scale: float = 1.
        n_gaussians_per_surface_triangle: int = 1
        keep_track_of_knn: bool = False
        knn_to_track: int = 16
        beta_mode: str = "average"  # 'learnable', 'average', 'weighted_average'
        primitive_types: str = "diamond"    # 'diamond', 'square'
        surface_mesh_to_bind_path: str = ""     # path of Open3D mesh
        learn_surface_mesh_positions: bool = True
        learn_surface_mesh_opacity: bool = True
        learn_surface_mesh_scales: bool = True
        freeze_gaussians: bool = False
        spatial_lr_scale: float = 10.
        spatial_extent: float = 3.5
        color_clip: Any = 2.0
        
        gs_color_inherit_vertices: bool = True
        init_gs_opacity: float = 0.5

        geometry_convert_from: str = ""
        
        # For texture mesh extraction
        square_size_in_texture: int = 10

        pred_normal: bool = False

    cfg: Config

    def configure(self, o3d_mesh=None) -> None:
        super().configure()
        self.active_sh_degree = 0
        self.sh_levels = self.cfg.sh_levels
        # Color features
        self._sh_coordinates_dc = torch.empty(0)
        self._sh_coordinates_rest = torch.empty(0)

        if self.cfg.surface_mesh_to_bind_path is not None:
            self.binded_to_surface_mesh = True
            self.load_surface_mesh_to_bind(o3d_mesh)
        else:
            self.binded_to_surface_mesh = False
            self._points = torch.empty(0)

        self.n_gaussians_per_surface_triangle = self.cfg.n_gaussians_per_surface_triangle

        self.knn_dists = None
        self.knn_idx = None

        self.prepare_primitive_polygon()

        # Texture attributes
        self._texture_initialized = False
        self.verts_uv, self.faces_uv = None, None
        
        
        if self.binded_to_surface_mesh and not self.learn_opacities:
            all_densities = inverse_sigmoid(
                0.9999 * torch.ones((self.n_points, 1), dtype=torch.float, device=self.device)
            )
        else:
            all_densities = inverse_sigmoid(
                self.cfg.init_gs_opacity * torch.ones((self.n_points, 1), dtype=torch.float, device=self.device)
            )
        self.all_densities = nn.Parameter(all_densities, requires_grad=self.learn_opacities)
        self.return_one_densities = False

        # Beta mode
        if self.cfg.beta_mode == "learnable":
            self._log_beta = torch.empty(0)

        
        self.initialize_learnable_radiuses()
        # self.update_texture_features()
        self.training_setup()


    def prune_isolated_points(self, verts, faces, vert_colors):
        orn = get_one_ring_neighbors(faces)
        vert_idx_kept = list(orn.keys())
        new_vert_idx = np.arange(len(vert_idx_kept), dtype=int)
        mapping_old2new = dict(zip(vert_idx_kept, new_vert_idx))

        new_verts = verts[vert_idx_kept]
        new_vert_colors = vert_colors[vert_idx_kept]
        new_faces = np.zeros_like(faces)
        for i in range(len(faces)):
            for j in range(3):
                new_faces[i, j] = mapping_old2new[faces[i, j]]
        return new_verts, new_faces, new_vert_colors
        
    def create_from_3dgs(self):
        ...
    

    def load_surface_mesh_to_bind(self, mesh=None):
        self.binded_to_surface_mesh = True
        self.learn_positions = self.cfg.learn_surface_mesh_positions
        self.learn_scales = self.cfg.learn_surface_mesh_scales
        self.learn_quaternions = self.cfg.learn_surface_mesh_scales
        self.learn_opacities = self.cfg.learn_surface_mesh_opacity

        # Load mesh with open3d
        threestudio.info(f"Loading mesh to bind from: {self.cfg.surface_mesh_to_bind_path}...")
        if mesh is None:
            o3d_mesh = o3d.io.read_triangle_mesh(self.cfg.surface_mesh_to_bind_path)
        else:
            o3d_mesh = mesh

        # self._surface_mesh_faces = nn.Parameter(
        #     torch.as_tensor(
        #         np.array(o3d_mesh.triangles), device=self.device
        #     ).requires_grad_(False)
        # )
        verts = np.array(o3d_mesh.vertices)
        faces = np.array(o3d_mesh.triangles)
        vert_colors = np.array(o3d_mesh.vertex_colors)
        if len(vert_colors) == 0:
            vert_colors = np.ones_like(verts) * 0.5
        verts, faces, vert_colors = self.prune_isolated_points(
            verts, faces, vert_colors
        )
        self.register_buffer(
            "_surface_mesh_faces", torch.as_tensor(faces, device=self.device)
        )
        surface_mesh_thickness = self.cfg.spatial_extent / 1_000_000
        self.surface_mesh_thickness = nn.Parameter(
            torch.as_tensor(
                surface_mesh_thickness, dtype=torch.float32, device=self.device
            ).requires_grad_(False)
        )

        threestudio.info("Binding radiance cloud to surface mesh...")
        self.set_surface_triangle_bary_coords(self.cfg.n_gaussians_per_surface_triangle)
        self._points = nn.Parameter(
            torch.as_tensor(
                verts, dtype=torch.float32, device=self.device
            ).requires_grad_(self.learn_positions)
        )
        self._n_points = len(faces) * self.cfg.n_gaussians_per_surface_triangle
        self._vertex_colors = torch.as_tensor(
            vert_colors, dtype=torch.float32, device=self.device
        )
        
        if self.cfg.gs_color_inherit_vertices:
            faces_colors = self._vertex_colors[self._surface_mesh_faces]  # n_faces, 3, n_coords
            colors = faces_colors[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_colors
            colors = colors.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_colors
            colors = colors.reshape(-1, 3)  # n_faces * n_gaussians_per_face, n_colors

            # Initialize color features
            sh_coordinates_dc = RGB2SH(colors).unsqueeze(dim=1)
        else:
            sh_coordinates_dc = RGB2SH(
                0.5 * torch.ones(self._n_points, 1, 3)
            )
        self._sh_coordinates_dc = nn.Parameter(
            sh_coordinates_dc.to(self.device, dtype=torch.float32), 
            requires_grad=(not self.cfg.freeze_gaussians)
        )
        self._sh_coordinates_rest = nn.Parameter(
            torch.zeros(self._n_points, self.sh_levels**2-1, 3).to(self.device),
            requires_grad=(not self.cfg.freeze_gaussians)
        )


    def set_surface_triangle_bary_coords(self, n_gaussians_per_surface_triangle: int) -> None:
        if n_gaussians_per_surface_triangle == 1:
                self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/3, 1/3, 1/3]],
                    dtype=torch.float32,
                    device=self.device,
                )[..., None]
            
        if n_gaussians_per_surface_triangle == 3:
            self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]],
                dtype=torch.float32,
                device=self.device,
            )[..., None]
        
        if n_gaussians_per_surface_triangle == 4:
            self.surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3],
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3]],
                dtype=torch.float32,
                device=self.device,
            )[..., None]  # n_gaussians_per_face, 3, 1
            
        if n_gaussians_per_surface_triangle == 6:
            self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
                [1/6, 5/12, 5/12],
                [5/12, 1/6, 5/12],
                [5/12, 5/12, 1/6]],
                dtype=torch.float32,
                device=self.device,
            )[..., None]

    def prepare_primitive_polygon(self):
        self._diamond_verts = torch.Tensor(
                [[0., -1., 0.], [0., 0, 1.], 
                [0., 1., 0.], [0., 0., -1.]]
                ).to(self.device)
        self._square_verts = torch.Tensor(
                [[0., -1., 1.], [0., 1., 1.], 
                [0., 1., -1.], [0., -1., -1.]]
                ).to(self.device)
        if self.cfg.primitive_types == 'diamond':
            self.primitive_verts = self._diamond_verts  # Shape (n_vertices_per_gaussian, 3)
        elif self.cfg.primitive_types == 'square':
            self.primitive_verts = self._square_verts  # Shape (n_vertices_per_gaussian, 3)
        self.primitive_triangles = torch.Tensor(
            [[0, 2, 1], [0, 3, 2]]
            ).to(self.device).long()  # Shape (n_triangles_per_gaussian, 3)
        self.primitive_border_edges = torch.Tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]]
            ).to(self.device).long()  # Shape (n_edges_per_gaussian, 2)
        self.n_vertices_per_gaussian = len(self.primitive_verts)
        self.n_triangles_per_gaussian = len(self.primitive_triangles)
        self.n_border_edges_per_gaussian = len(self.primitive_border_edges)

    def initialize_learnable_radiuses(self):
        self.scale_activation = torch.exp
        self.scale_inverse_activation = torch.log
        if self.binded_to_surface_mesh:
            # First gather vertices of all triangles
            faces_verts = self._points[self._surface_mesh_faces]  # n_faces, 3, n_coords
            
            # Then, compute initial scales
            scales = (faces_verts - faces_verts[:, [1, 2, 0]]).norm(dim=-1).min(dim=-1)[0] * self.surface_triangle_circle_radius
            scales = scales.clamp_min(0.0000001).reshape(len(faces_verts), -1, 1).expand(-1, self.cfg.n_gaussians_per_surface_triangle, 2).clone().reshape(-1, 2)
            self._scales = nn.Parameter(
                self.scale_inverse_activation(scales),
                requires_grad=self.learn_scales
            )
            
            # We actually don't learn quaternions here, but complex numbers to encode a 2D rotation in the triangle's plane
            complex_numbers = torch.zeros(self._n_points, 2).to(self.device)
            complex_numbers[:, 0] = 1.
            self._quaternions = nn.Parameter(
                complex_numbers,
                requires_grad=self.learn_quaternions
            )
        else:
            raise NotImplementedError
        
    def training_setup(self):
        training_args = self.cfg
        self.spatial_lr_scale = self.cfg.spatial_lr_scale    # TODO: put it here for now

        l = []
        if self.binded_to_surface_mesh:
            if self.learn_positions:
                l += [
                    {
                        "params": [self._points],
                        "lr": C(training_args.position_lr, 0, 0) * self.spatial_lr_scale,
                        "name": "points",
                    },
                ]
            if not self.cfg.freeze_gaussians:
                l += [
                    {
                        "params": [self._sh_coordinates_dc],
                        "lr": C(training_args.feature_lr, 0, 0),
                        "name": "f_dc",
                    },
                    {
                        "params": [self._sh_coordinates_rest],
                        "lr": C(training_args.feature_lr, 0, 0) / 20.0,
                        "name": "f_rest",
                    }
                ]
            if self.learn_opacities:
                l += [
                    {
                        "params": [self.all_densities],
                        "lr": C(training_args.opacity_lr, 0, 0),
                        "name": "all_densities",
                    }
                ]
            if self.learn_scales:
                l += [
                    {
                        "params": [self._scales],
                        "lr": C(training_args.scaling_lr, 0, 0),
                        "name": "scales"
                    },
                    {
                        "params": [self._quaternions],
                        "lr": C(training_args.rotation_lr, 0, 0),
                        "name": "quaternions"
                    }
                ]
        else:
            raise NotImplementedError
        
        self.optimize_list = l
        self.optimize_params = [d["name"] for d in l]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.color_clip = C(self.cfg.color_clip, 0, 0)

    def update_learning_rate(self, iteration):
        """Following SuGaR source code, only update position lr here"""
        for param_group in self.optimizer.param_groups:
            if not ("name" in param_group):
                continue
            if param_group["name"] == "points":
                param_group["lr"] = C(
                    self.cfg.position_lr, 0, iteration, interpolation="exp"
                ) * self.spatial_lr_scale
            if param_group["name"] == "f_dc":
                param_group["lr"] = C(
                    self.cfg.feature_lr, 0, iteration, interpolation="exp"
                )
            if param_group["name"] == "f_rest":
                param_group["lr"] = C(
                    self.cfg.feature_lr, 0, iteration, interpolation="exp"
                ) / 20.0

        self.color_clip = C(self.cfg.color_clip, 0, iteration)
    
    def merge_optimizer(self, net_optimizer):
        l = self.optimize_list
        for param in net_optimizer.param_groups:
            l.append(
                {
                    "params": param["params"],
                    "lr": param["lr"],
                }
            )
        self.optimizer = torch.optim.AdamW(l, lr=0.0, betas=[0.9, 0.99], eps=1.e-15)
        return self.optimizer
                
    @property
    def n_points(self):
        if not self.binded_to_surface_mesh:
            return len(self._points)
        else:
            return self._n_points
        
    @property
    def n_verts(self):
        if self.binded_to_surface_mesh:
            return len(self._points)
        else:
            raise ValueError("No verts when with no mesh binded!")
    
    @property
    def n_faces(self):
        if self.binded_to_surface_mesh:
            return len(self._surface_mesh_faces)
        else:
            raise ValueError("No faces when with no mesh binded!")
        
    @property
    def points(self):
        if not self.binded_to_surface_mesh:
            if (not self.learnable_positions) and self.learnable_shifts:
                return self._points + self.max_shift * 2 * (torch.sigmoid(self.shifts) - 0.5)
            else:
                return self._points
        else:
            # First gather vertices of all triangles
            faces_verts = self._points[self._surface_mesh_faces]  # n_faces, 3, n_coords
            
            # Then compute the points using barycenter coordinates in the surface triangles
            points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
            points = points.sum(dim=-2)  # n_faces, n_gaussians_per_face, n_coords
            
            return points.reshape(self._n_points, 3)  # n_faces * n_gaussians_per_face, n_coords

    @property
    def sh_coordinates(self):
        features_dc = self._sh_coordinates_dc
        features_dc = features_dc.clip(-self.color_clip, self.color_clip)
        features_rest = self._sh_coordinates_rest
        return torch.cat([features_dc, features_rest], dim=1)

    @property
    def _texture_features(self):    # starts with "_" to avoid update texture features in update step
        if not self._texture_initialized:
            self.update_texture_features(self.cfg.square_size_in_texture)
        return self.sh_coordinates[self.point_idx_per_pixel]
    
    @property
    def strengths(self):
        return torch.sigmoid(self.all_densities.view(-1, 1))
    
    @property
    def radiuses(self):
        return torch.cat([self._quaternions, self._scales], dim=-1)[None]
    
    @property
    def scaling(self):
        if not self.binded_to_surface_mesh:
            scales = self.scale_activation(self._scales)
        else:
            scales = torch.cat([
                self.surface_mesh_thickness * torch.ones(len(self._scales), 1, device=self.device), 
                self.scale_activation(self._scales)
                ], dim=-1)
        return scales
    
    @property
    def quaternions(self):
        if not self.binded_to_surface_mesh:
            quaternions = self._quaternions
        else:
            # We compute quaternions to enforce face normals to be the first axis of gaussians
            R_0 = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1)

            # We use the first side of every triangle as the second base axis
            faces_verts = self._points[self._surface_mesh_faces]
            base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)

            # We use the cross product for the last base axis
            base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
            
            # We now apply the learned 2D rotation to the base quaternion
            complex_numbers = torch.nn.functional.normalize(self._quaternions, dim=-1).view(len(self._surface_mesh_faces), self.cfg.n_gaussians_per_surface_triangle, 2)
            R_1 = complex_numbers[..., 0:1] * base_R_1[:, None] + complex_numbers[..., 1:2] * base_R_2[:, None]
            R_2 = -complex_numbers[..., 1:2] * base_R_1[:, None] + complex_numbers[..., 0:1] * base_R_2[:, None]

            # We concatenate the three vectors to get the rotation matrix
            R = torch.cat([R_0[:, None, ..., None].expand(-1, self.cfg.n_gaussians_per_surface_triangle, -1, -1).clone(),
                        R_1[..., None],
                        R_2[..., None]],
                        dim=-1).view(-1, 3, 3)
            quaternions = matrix_to_quaternion(R)
            
        return torch.nn.functional.normalize(quaternions, dim=-1)
    
    @property
    def get_face_normals(self) -> Float[Tensor, "N_faces 3"]:
        return F.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1)
    
    @property
    def get_gs_normals(self) -> Float[Tensor, "N_gs 3"]:
        return self.get_face_normals.repeat_interleave(self.cfg.n_gaussians_per_surface_triangle, dim=0)
    
    @property
    def get_features(self):
        return self.sh_coordinates

    @property
    def get_scaling(self):
        return self.scaling
    
    @property
    def get_opacity(self):
        return self.strengths
    
    @property
    def get_rotation(self):
        return self.quaternions
    
    @property
    def get_xyz(self):
        return self.points
    
    @property
    def get_xyz_verts(self):
        if self.binded_to_surface_mesh:
            return self._points
        else:
            raise ValueError
        
    @property
    def get_faces(self):
        if self.binded_to_surface_mesh:
            return self._surface_mesh_faces
        else:
            raise ValueError
    
    @property
    def _mesh(self):    # starts with "_" to avoid update texture features in update step
        textures_uv = TexturesUV(
            maps=SH2RGB(self._texture_features[..., 0, :][None]),
            verts_uvs=self.verts_uv[None],
            faces_uvs=self.faces_uv[None],
            sampling_mode='nearest',
            )
        
        return Meshes(
            verts=[self.triangle_vertices],   
            faces=[self.triangles],
            textures=textures_uv,
        )
    
    @property
    def surface_mesh(self):
        # Create a Meshes object
        surface_mesh = Meshes(
            verts=[self._points.to(self.device)],   
            faces=[self._surface_mesh_faces.to(self.device)],
            textures=TexturesVertex(verts_features=self._vertex_colors[None].clamp(0, 1).to(self.device)),
            # verts_normals=[verts_normals.to(rc.device)],
        )
        return surface_mesh
    
    def get_covariance(self, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
        scaling = self.scaling
        if inverse_scales:
            scaling = 1. / scaling.clamp(min=1e-8)
        scaled_rotation = quaternion_to_matrix(self.quaternions) * scaling[:, None]
        if return_sqrt:
            return scaled_rotation

        cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
        if return_full_matrix:
            return cov3Dmatrix

        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)
        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]

        return cov3D
    
    def update_texture_features(self, square_size_in_texture=2):
        features = self.sh_coordinates.view(len(self.points), -1)
        faces_uv, verts_uv, texture_img, point_idx_per_pixel = _convert_vertex_colors_to_texture(
            self, 
            features,
            square_size=square_size_in_texture,
            )
        self.texture_size = texture_img.shape[0]
        self.verts_uv = verts_uv
        self.faces_uv = faces_uv
        texture_img = texture_img.view(self.texture_size, self.texture_size, -1, 3)

        self.point_idx_per_pixel = torch.nn.Parameter(point_idx_per_pixel, requires_grad=False)
        self._texture_initialized = True

    def reset_neighbors(self, knn_to_track:int=None):
        if not hasattr(self, 'knn_to_track'):
            if knn_to_track is None:
                knn_to_track = 16
            self.knn_to_track = knn_to_track
        else:
            if knn_to_track is None:
                knn_to_track = self.knn_to_track 
        # Compute KNN               
        with torch.no_grad():
            self.knn_to_track = knn_to_track
            knns = knn_points(self.points[None], self.points[None], K=knn_to_track)
            self.knn_dists = knns.dists[0]
            self.knn_idx = knns.idx[0]

    def get_points_rgb(
        self,
        camera_centers: Float[Tensor, "N_pts 3"] | Float[Tensor, "1 3"] = None,
    ) -> Float[Tensor, "N_pts 3"]:
        positions = self.points
        sh_levels = self.cfg.sh_levels

        if sh_levels == 1:
            colors = SH2RGB(self._sh_coordinates_dc).view(-1, 3)
        else:
            if camera_centers is not None:
                render_directions = F.normalize(positions - camera_centers, dim=-1)
            else:
                raise ValueError("camera_centers must be provided.")
            
            sh_coordinates = self.sh_coordinates[:, :sh_levels**2]

            shs_view = sh_coordinates.transpose(-1, -2).view(-1, 3, sh_levels**2)
            sh2rgb = eval_sh(sh_levels-1, shs_view, render_directions)
            colors = torch.clamp_min(sh2rgb + 0.5, 0.0).view(-1, 3)
            
        return colors  
    
    """surface levels computation functions, not used"""
    def compute_surface_levels_point_clouds(self):
        ...

    # Named `compute_level_surface_points_from_camera_fast` in original code
    def compute_level_surface_points_from_camera(
        self,
        nerf_cameras,   # TODO: Wrapped p3d camera list
    ):
        ...


def _convert_vertex_colors_to_texture(
    rc: SuGaRModel, 
    colors: torch.Tensor,
    square_size: int=4,
):
    points_to_mesh = rc.points
    
    n_square_per_axis = int(np.sqrt(len(points_to_mesh)) + 1)
    texture_size = square_size * n_square_per_axis
    
    n_features = colors.shape[-1]
    
    point_idx_per_pixel = torch.zeros(texture_size, texture_size, device=rc.device).int()

    with torch.no_grad():
        # Build face UVs
        faces_uv = torch.Tensor(
            [[0, 2, 1], [0, 3, 2]]
            ).to(rc.device)[None] + 4 * torch.arange(len(points_to_mesh), device=rc.device)[:, None, None]
        faces_uv = faces_uv.view(-1, 3).long()

        # Build verts UVs
        verts_coords = torch.cartesian_prod(
            torch.arange(n_square_per_axis, device=rc.device), 
            torch.arange(n_square_per_axis, device=rc.device)
            )[:, None] * square_size
        verts_uv = torch.Tensor(
            [[1., 1.], [1., square_size-1], [square_size-1, square_size-1], [square_size-1, 1.]]
            ).to(rc.device)[None] + verts_coords
        verts_uv = verts_uv.view(-1, 2).long()[:4*len(points_to_mesh)] / texture_size

        # Build texture image
        texture_img = torch.zeros(texture_size, texture_size, n_features, device=rc.device)    
        n_squares_filled = 0
        for i in range(n_square_per_axis):
            for j in range(n_square_per_axis):
                if n_squares_filled >= len(points_to_mesh):
                    break
                start_idx_i = i * square_size
                start_idx_j = j * square_size
                texture_img[..., 
                            start_idx_i:start_idx_i + square_size, 
                            start_idx_j:start_idx_j + square_size, :] = colors[i*n_square_per_axis + j].unsqueeze(0).unsqueeze(0)
                point_idx_per_pixel[...,
                                    start_idx_i:start_idx_i + square_size, 
                                    start_idx_j:start_idx_j + square_size] = i*n_square_per_axis + j
                n_squares_filled += 1
                
        texture_img = texture_img.transpose(0, 1)
        texture_img = texture_img.flip(0)
        
        point_idx_per_pixel = point_idx_per_pixel.transpose(0, 1)
        point_idx_per_pixel = point_idx_per_pixel.flip(0)
    
    return faces_uv, verts_uv, texture_img, point_idx_per_pixel


# ============= Spherical Harmonics ============ #
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


from ..utils.sugar_utils import getProjectionMatrix, getWorld2View2, fov2focal
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

class GSCamera(torch.nn.Module):
    """Class to store Gaussian Splatting camera parameters.
    """
    def __init__(self, R, T, FoVx, FoVy, 
                 image_height, image_width,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, 
                 data_device = "cuda",
                 ):
        """
        Args:
            colmap_id (int): ID of the camera in the COLMAP reconstruction.
            R (np.array): Rotation matrix.
            T (np.array): Translation vector.
            FoVx (float): Field of view in the x direction.
            FoVy (float): Field of view in the y direction.
            image (np.array): GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(GSCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_height = image_height
        self.image_width = image_width

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    @property
    def device(self):
        return self.world_view_transform.device
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self
    
def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    R = torch.Tensor(np.array([gs_camera.R for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    p0_pytorch3d = (
        -(
            torch.Tensor(
                (cx[0], cy[0]),
            )[
                None
            ].to(device)
            - c0
        )
        / scale
    )
    focal_pytorch3d = (
        torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    )
    K = _get_sfm_calibration_matrix(
        1, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    K = K.expand(N, -1, -1)

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras