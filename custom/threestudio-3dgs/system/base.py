import os
import random
from dataclasses import dataclass, field

import numpy as np
import threestudio
import torch
import torch.nn.functional as F
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer, parse_scheduler
from threestudio.utils.loss import tv_loss
from threestudio.utils.ops import get_cam_info_gaussian
from threestudio.utils.typing import *
from torch.cuda.amp import autocast
from torchmetrics import PearsonCorrCoef
import open3d as o3d

from ..geometry.gaussian_base import BasicPointCloud, Camera, SH2RGB, RGB2SH
from ..geometry.sugar import SuGaRModel, GSCamera, convert_camera_from_gs_to_pytorch3d


from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer import (
    AmbientLights,
    MeshRenderer,
    SoftPhongShader,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.io import save_obj



class BaseSuGaRSystem(BaseLift3DSystem):

    @dataclass
    class Config(BaseLift3DSystem.Config):
        postprocess: bool = False
        postprocess_density_threshold: float = 0.1
        postprocess_iterations: int = 5
        square_size_in_texture: int = 10
        export_resolution: int = 512
        
    cfg: Config
    geometry: SuGaRModel

    @torch.no_grad()
    def export_mesh_to_ply(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            self.geometry.get_xyz_verts.detach().cpu().numpy()
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            self.geometry.get_faces.detach().cpu().numpy()
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            self.geometry._vertex_colors.detach().cpu().numpy()
        )
        mesh.compute_vertex_normals()

        mesh_save_path = os.path.join(
            self.get_save_dir(), f"exported_mesh_step{self.global_step}.ply"
        )
        o3d.io.write_triangle_mesh(
            mesh_save_path, mesh, write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True
        )
        threestudio.info(f"The current mesh is exported to {mesh_save_path}.")
    
    # For texture mesh extraction
    def on_predict_start(self) -> None:
        if self.cfg.postprocess:
            self.postprocess_mesh()

        rc = self.geometry
        rc.update_texture_features(self.cfg.square_size_in_texture)
        # square_size = self.cfg.square_size
        square_size = self.cfg.square_size_in_texture

        surface_mesh = rc.surface_mesh
        verts = surface_mesh.verts_list()[0]
        faces = surface_mesh.faces_list()[0]
        faces_verts = verts[faces]

        n_triangles = len(faces)
        n_gaussians_per_triangle = rc.n_gaussians_per_surface_triangle
        n_squares = n_triangles // 2 + 1
        n_square_per_axis = int(np.sqrt(n_squares) + 1)
        texture_size = square_size * (n_square_per_axis)

        n_sh = 1
        faces_features = rc.sh_coordinates[:, :n_sh].reshape(n_triangles, n_gaussians_per_triangle, n_sh * 3)
        n_features = faces_features.shape[-1]

        self.faces_uv = torch.arange(3 * n_triangles, device=rc.device).view(n_triangles, 3)

        # Build corresponding vertices UV
        vertices_uv = torch.cartesian_prod(
            torch.arange(n_square_per_axis, device=rc.device),
            torch.arange(n_square_per_axis, device=rc.device))
        bottom_verts_uv = torch.cat(
            [vertices_uv[n_square_per_axis:-1, None], vertices_uv[:-n_square_per_axis - 1, None],
            vertices_uv[n_square_per_axis + 1:, None]],
            dim=1)
        top_verts_uv = torch.cat(
            [vertices_uv[1:-n_square_per_axis, None], vertices_uv[:-n_square_per_axis - 1, None],
            vertices_uv[n_square_per_axis + 1:, None]],
            dim=1)

        vertices_uv = torch.cartesian_prod(
            torch.arange(n_square_per_axis, device=rc.device),
            torch.arange(n_square_per_axis, device=rc.device))[:, None]
        u_shift = torch.tensor([[1, 0]], dtype=torch.int32, device=rc.device)[:, None]
        v_shift = torch.tensor([[0, 1]], dtype=torch.int32, device=rc.device)[:, None]
        bottom_verts_uv = torch.cat(
            [vertices_uv + u_shift, vertices_uv, vertices_uv + u_shift + v_shift],
            dim=1)
        top_verts_uv = torch.cat(
            [vertices_uv + v_shift, vertices_uv, vertices_uv + u_shift + v_shift],
            dim=1)

        self.verts_uv = torch.cat([bottom_verts_uv, top_verts_uv], dim=1)
        self.verts_uv = self.verts_uv * square_size
        self.verts_uv[:, 0] = self.verts_uv[:, 0] + torch.tensor([[-2, 1]], device=rc.device)
        self.verts_uv[:, 1] = self.verts_uv[:, 1] + torch.tensor([[2, 1]], device=rc.device)
        self.verts_uv[:, 2] = self.verts_uv[:, 2] + torch.tensor([[-2, -3]], device=rc.device)
        self.verts_uv[:, 3] = self.verts_uv[:, 3] + torch.tensor([[1, -1]], device=rc.device)
        self.verts_uv[:, 4] = self.verts_uv[:, 4] + torch.tensor([[1, 3]], device=rc.device)
        self.verts_uv[:, 5] = self.verts_uv[:, 5] + torch.tensor([[-3, -1]], device=rc.device)
        self.verts_uv = self.verts_uv.reshape(-1, 2) / texture_size

        # ---Build texture image
        # Start by computing pixel indices for each triangle
        self.texture_img = torch.zeros(texture_size, texture_size, n_features, device=rc.device)
        pixel_idx_inside_bottom_triangle = torch.zeros(0, 2, dtype=torch.int32, device=rc.device)
        pixel_idx_inside_top_triangle = torch.zeros(0, 2, dtype=torch.int32, device=rc.device)
        for tri_i in range(0, square_size - 1):
            for tri_j in range(0, tri_i + 1):
                pixel_idx_inside_bottom_triangle = torch.cat(
                    [pixel_idx_inside_bottom_triangle, torch.tensor([[tri_i, tri_j]], dtype=torch.int32, device=rc.device)],
                    dim=0)
        for tri_i in range(0, square_size):
            for tri_j in range(tri_i + 1, square_size):
                pixel_idx_inside_top_triangle = torch.cat(
                    [pixel_idx_inside_top_triangle, torch.tensor([[tri_i, tri_j]], dtype=torch.int32, device=rc.device)],
                    dim=0)

        bottom_triangle_pixel_idx = torch.cartesian_prod(
            torch.arange(n_square_per_axis, device=rc.device),
            torch.arange(n_square_per_axis, device=rc.device))[:, None] * square_size + pixel_idx_inside_bottom_triangle[
                                        None]
        top_triangle_pixel_idx = torch.cartesian_prod(
            torch.arange(n_square_per_axis, device=rc.device),
            torch.arange(n_square_per_axis, device=rc.device))[:, None] * square_size + pixel_idx_inside_top_triangle[None]
        triangle_pixel_idx = torch.cat(
            [bottom_triangle_pixel_idx[:, None],
            top_triangle_pixel_idx[:, None]],
            dim=1).view(-1, bottom_triangle_pixel_idx.shape[-2], 2)[:n_triangles]

        # Then we compute the barycentric coordinates of each pixel inside its corresponding triangle
        bottom_triangle_pixel_bary_coords = pixel_idx_inside_bottom_triangle.clone().float()
        bottom_triangle_pixel_bary_coords[..., 0] = -(bottom_triangle_pixel_bary_coords[..., 0] - (square_size - 2))
        bottom_triangle_pixel_bary_coords[..., 1] = (bottom_triangle_pixel_bary_coords[..., 1] - 1)
        bottom_triangle_pixel_bary_coords = (bottom_triangle_pixel_bary_coords + 0.) / (square_size - 3)
        bottom_triangle_pixel_bary_coords = torch.cat(
            [1. - bottom_triangle_pixel_bary_coords.sum(dim=-1, keepdim=True), bottom_triangle_pixel_bary_coords],
            dim=-1)
        top_triangle_pixel_bary_coords = pixel_idx_inside_top_triangle.clone().float()
        top_triangle_pixel_bary_coords[..., 0] = (top_triangle_pixel_bary_coords[..., 0] - 1)
        top_triangle_pixel_bary_coords[..., 1] = -(top_triangle_pixel_bary_coords[..., 1] - (square_size - 1))
        top_triangle_pixel_bary_coords = (top_triangle_pixel_bary_coords + 0.) / (square_size - 3)
        top_triangle_pixel_bary_coords = torch.cat(
            [1. - top_triangle_pixel_bary_coords.sum(dim=-1, keepdim=True), top_triangle_pixel_bary_coords],
            dim=-1)
        triangle_pixel_bary_coords = torch.cat(
            [bottom_triangle_pixel_bary_coords[None],
            top_triangle_pixel_bary_coords[None]],
            dim=0)  # 2, n_pixels_per_triangle, 3

        all_triangle_bary_coords = triangle_pixel_bary_coords[None].expand(n_squares, -1, -1, -1).reshape(-1,
                                                                                                        triangle_pixel_bary_coords.shape[
                                                                                                            -2], 3)
        all_triangle_bary_coords = all_triangle_bary_coords[:len(faces_verts)]

        pixels_space_positions = (all_triangle_bary_coords[..., None] * faces_verts[:, None]).sum(dim=-2)[:, :, None]

        gaussian_centers = rc.points.reshape(-1, 1, rc.n_gaussians_per_surface_triangle, 3)
        gaussian_inv_scaled_rotation = rc.get_covariance(return_full_matrix=True, return_sqrt=True,
                                                        inverse_scales=True).reshape(-1, 1,
                                                                                    rc.n_gaussians_per_surface_triangle,
                                                                                    3, 3)

        # Compute the density field as a sum of local gaussian opacities
        shift = (pixels_space_positions - gaussian_centers)
        warped_shift = gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = torch.exp(-1. / 2 * neighbor_opacities)  # / rc.n_gaussians_per_surface_triangle

        pixel_features = faces_features[:, None].expand(-1, neighbor_opacities.shape[1], -1, -1).gather(
            dim=-2,
            index=neighbor_opacities[..., None].argmax(dim=-2, keepdim=True).expand(-1, -1, -1, 3)
        )[:, :, 0, :]

        # pixel_alpha = neighbor_opacities.sum(dim=-1, keepdim=True)
        self.texture_img[(triangle_pixel_idx[..., 0], triangle_pixel_idx[..., 1])] = pixel_features

        self.texture_img = self.texture_img.transpose(0, 1)
        self.texture_img = SH2RGB(self.texture_img.flip(0)) # TODO: Need to be check
        # self.texture_img = self.texture_img.flip(0)

        faces_per_pixel = 1
        max_faces_per_bin = 50_000
        mesh_raster_settings = RasterizationSettings(
            image_size=(self.cfg.export_resolution, self.cfg.export_resolution),
            blur_radius=0.0,
            faces_per_pixel=faces_per_pixel,
            # max_faces_per_bin=max_faces_per_bin
        )
        lights = AmbientLights(device=rc.device)
        self.mesh_rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=mesh_raster_settings,
        )
        self.mesh_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(
                device=rc.device,
                cameras=None,
                lights=lights,
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
            )
        )
        texture_idx = torch.cartesian_prod(
            torch.arange(texture_size, device=rc.device),
            torch.arange(texture_size, device=rc.device)
        ).reshape(texture_size, texture_size, 2
                )
        texture_idx = torch.cat([texture_idx, torch.zeros_like(texture_idx[..., 0:1])], dim=-1)
        self.texture_counter = torch.zeros(texture_size, texture_size, 1, device=self.device)
        idx_textures_uv = TexturesUV(
            maps=texture_idx[None].float(),  # self.texture_img[None]),
            verts_uvs=self.verts_uv[None],
            faces_uvs=self.faces_uv[None],
            sampling_mode='nearest',
        )
        self.idx_mesh = Meshes(
            verts=[rc.surface_mesh.verts_list()[0]],
            faces=[rc.surface_mesh.faces_list()[0]],
            textures=idx_textures_uv,
        )

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        batch_size = batch["c2w"].shape[0]
        assert batch_size == 1, "Now only support batch size = 1"
        out = self.forward(batch, compute_color_in_rasterizer=True)
        rgb_img = out["comp_rgb"][0]
        
        c2w = batch["c2w"][0].cpu().numpy()
        Rt = np.linalg.inv(c2w)
        T = Rt[:3, 3]
        R = Rt[:3, :3].transpose()
        fov = batch["fovy"].cpu().numpy()[0]
        height = batch["height"]
        width = batch["width"]
        gs_camera = GSCamera(
            R=R, T=T, FoVx=fov, FoVy=fov,
            image_height=height,
            image_width=width,
            data_device="cuda"
        )
        p3d_camera = convert_camera_from_gs_to_pytorch3d(gs_cameras=[gs_camera])[0]

        fragments = self.mesh_renderer.rasterizer(self.idx_mesh, cameras=p3d_camera)
        idx_img = self.mesh_renderer.shader(fragments, self.idx_mesh, cameras=p3d_camera)[0, ..., :2]

        update_mask = fragments.zbuf[0, ..., 0] > 0
        idx_to_update = idx_img[update_mask].round().long()

        use_average = True
        if not use_average:
            self.texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = rgb_img[update_mask]
        else:
            no_initialize_mask = self.texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])][..., 0] != 0
            self.texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = no_initialize_mask[..., None] * self.texture_img[
                (idx_to_update[..., 0], idx_to_update[..., 1])]

            self.texture_img[(idx_to_update[..., 0], idx_to_update[..., 1])] = self.texture_img[(
                idx_to_update[..., 0], idx_to_update[..., 1])] + rgb_img[update_mask]
            self.texture_counter[(idx_to_update[..., 0], idx_to_update[..., 1])] = self.texture_counter[(
                idx_to_update[..., 0], idx_to_update[..., 1])] + 1
            
    def on_predict_epoch_end(self) -> None:
        self.texture_img = self.texture_img / self.texture_counter.clamp(min=1)

        textures_uv = TexturesUV(
            maps=self.texture_img[None],
            verts_uvs=self.verts_uv[None],
            faces_uvs=self.faces_uv[None],
            sampling_mode='nearest',
        )
        textured_mesh = Meshes(
            verts=[self.geometry.surface_mesh.verts_list()[0]],
            faces=[self.geometry.surface_mesh.faces_list()[0]],
            textures=textures_uv
        )
        
        threestudio.info("Texture extracted.")        
        threestudio.info("Saving textured mesh...")
        
        mesh_save_path = os.path.join(
            self.get_save_dir(), "extracted_mesh.obj"
        )
        with torch.no_grad():
            save_obj(  
                mesh_save_path,
                verts=textured_mesh.verts_list()[0],
                faces=textured_mesh.faces_list()[0],
                verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
                faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
                texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
            )

    @torch.no_grad()
    def postprocess_mesh(self,):
        threestudio.info("Postprocessing mesh by removing border triangles with low-opacity gaussians...")
        args = self.cfg
        postprocess_density_threshold = args.postprocess_density_threshold
        postprocess_iterations = args.postprocess_iterations

        refined_sugar = self.geometry

        new_verts = refined_sugar.surface_mesh.verts_list()[0].detach().clone()
        new_faces = refined_sugar.surface_mesh.faces_list()[0].detach().clone()
        new_normals = refined_sugar.surface_mesh.faces_normals_list()[0].detach().clone()
        
        # For each face, get the 3 edges
        edges0 = new_faces[..., None, (0,1)].sort(dim=-1)[0]
        edges1 = new_faces[..., None, (1,2)].sort(dim=-1)[0]
        edges2 = new_faces[..., None, (2,0)].sort(dim=-1)[0]
        all_edges = torch.cat([edges0, edges1, edges2], dim=-2)

        # We start by identifying the inside faces and border faces
        face_mask = refined_sugar.strengths[..., 0] > -1.
        for i in range(postprocess_iterations):
            threestudio.info("\nStarting postprocessing iteration", i)
            # We look for edges that appear in the list at least twice (their NN is themselves)
            edges_neighbors = knn_points(all_edges[face_mask].view(1, -1, 2).float(), all_edges[face_mask].view(1, -1, 2).float(), K=2)
            # If all edges of a face appear in the list at least twice, then the face is inside the mesh
            is_inside = (edges_neighbors.dists[0][..., 1].view(-1, 3) < 0.01).all(-1)
            # We update the mask by removing border faces
            face_mask[face_mask.clone()] = is_inside

        # We then add back border faces with high-density
        face_centers = new_verts[new_faces].mean(-2)
        face_densities = refined_sugar.compute_density(face_centers[~face_mask])
        face_mask[~face_mask.clone()] = face_densities > postprocess_density_threshold

        # And we create the new mesh and SuGaR model
        new_faces = new_faces[face_mask]
        new_normals = new_normals[face_mask]

        new_scales = refined_sugar._scales.reshape(len(face_mask), -1, 2)[face_mask].view(-1, 2)
        new_quaternions = refined_sugar._quaternions.reshape(len(face_mask), -1, 2)[face_mask].view(-1, 2)
        new_densities = refined_sugar.all_densities.reshape(len(face_mask), -1, 1)[face_mask].view(-1, 1)
        new_sh_coordinates_dc = refined_sugar._sh_coordinates_dc.reshape(len(face_mask), -1, 1, 3)[face_mask].view(-1, 1, 3)
        new_sh_coordinates_rest = refined_sugar._sh_coordinates_rest.reshape(len(face_mask), -1, 15, 3)[face_mask].view(-1, 15, 3)
        
        new_o3d_mesh = o3d.geometry.TriangleMesh()
        new_o3d_mesh.vertices = o3d.utility.Vector3dVector(new_verts.cpu().numpy())
        new_o3d_mesh.triangles = o3d.utility.Vector3iVector(new_faces.cpu().numpy())
        new_o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(new_normals.cpu().numpy())
        new_o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(torch.ones_like(new_verts).cpu().numpy())

        refined_sugar.configure(new_o3d_mesh)
        refined_sugar._scales[...] = new_scales
        refined_sugar._quaternions[...] = new_quaternions
        refined_sugar.all_densities[...] = new_densities
        refined_sugar._sh_coordinates_dc[...] = new_sh_coordinates_dc
        refined_sugar._sh_coordinates_rest[...] = new_sh_coordinates_rest
        threestudio.info("Mesh postprocessed.")


        



        
