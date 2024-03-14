from collections import defaultdict

import numpy as np
import torch
import math
from typing import NamedTuple
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
from pytorch3d.transforms import quaternion_apply, matrix_to_quaternion, quaternion_to_matrix
from ..geometry.gaussian_base import GaussianBaseModel

from threestudio.utils.typing import *

scale_activation = torch.exp
scale_inverse_activation = torch.log


def _initialize_radiuses_gauss_rasterizer(sugar):
    """Function to initialize the  of a SuGaR model.

    Args:
        sugar (SuGaR): SuGaR model.

    Returns:
        Tensor: Tensor with shape (n_points, 4+3) containing
            the initial quaternions and scaling factors.
    """
    # Initialize learnable radiuses
    # sugar.image_height = int(sugar.nerfmodel.training_cameras.height[0].item())
    # sugar.image_width = int(sugar.nerfmodel.training_cameras.width[0].item())
    #
    # all_camera_centers = sugar.nerfmodel.training_cameras.camera_to_worlds[..., 3]
    # all_camera_dists = torch.cdist(sugar.points, all_camera_centers)[None]
    # d_charac = all_camera_dists.mean(-1, keepdim=True)
    #
    # ndc_factor = 1.
    # sugar.min_ndc_radius = ndc_factor * 2. / min(sugar.image_height, sugar.image_width)
    # sugar.max_ndc_radius = ndc_factor * 2. * 0.05  # 2. * 0.01
    # sugar.min_radius = sugar.min_ndc_radius / sugar.focal_factor * d_charac
    # sugar.max_radius = sugar.max_ndc_radius / sugar.focal_factor * d_charac

    knn = knn_points(sugar.points[None], sugar.points[None], K=4)
    use_sqrt = True
    use_mean = False
    initial_radius_normalization = 1.  # 1., 0.1
    if use_sqrt:
        knn_dists = torch.sqrt(knn.dists[..., 1:])
    else:
        knn_dists = knn.dists[..., 1:]
    if use_mean:
        print("Use mean to initialize scales.")
        radiuses = knn_dists.mean(-1, keepdim=True).clamp_min(0.0000001) * initial_radius_normalization
    else:
        print("Use min to initialize scales.")
        radiuses = knn_dists.min(-1, keepdim=True)[0].clamp_min(0.0000001) * initial_radius_normalization

    res = inverse_radius_fn(radiuses=radiuses)
    sugar.radius_dim = res.shape[-1]

    return res

def get_one_ring_neighbors(faces) -> Dict[Int, List[Int]]:
    mapping = defaultdict(set)
    for f in faces:
        for j in range(3):  # for each vert in the face
            i, k = (j + 1) % 3, (j + 2) % 3 # get the 2 other vertices
            mapping[f[j]].add(f[i])
            mapping[f[j]].add(f[k])
    orn = {k: list(v) for k, v in mapping.items()}  # convert to list
    return orn

def inverse_radius_fn(radiuses: torch.Tensor):
    scales = scale_inverse_activation(radiuses.expand(-1, -1, 3).clone())
    quaternions = matrix_to_quaternion(
        torch.eye(3)[None, None].repeat(1, radiuses.shape[1], 1, 1).to(radiuses.device)
    )
    return torch.cat([quaternions, scales], dim=-1)


class SuGaRRegularizer():
    def __init__(
        self,
        gaussians: GaussianBaseModel,
        initialize: bool = True,
        keep_track_of_knn: bool = False,
        knn_to_track: int = 16,
        surface_mesh_to_bind=None,  # Open3D mesh
        beta_mode='average',  # 'learnable', 'average', 'weighted_average'
    ):
        """
        Args:
            gaussians (GaussianSplattingWrapper): A vanilla Gaussian Splatting model trained for 7k iterations.
            initialize (bool, optional): Whether to initialize the radiuses. Defaults to True.

            keep_track_of_knn (bool, optional): Whether to keep track of the KNN information for training regularization. Defaults to False.
            knn_to_track (int, optional): Number of KNN to track. Defaults to 16.
            surface_mesh_to_bind (None, optional): Surface mesh to bind the Gaussians to. Defaults to None.
            beta_mode (str, optional): Whether to use a learnable beta, or to average the beta values. Defaults to 'average'.

        """

        self.gaussians = gaussians
        # initialize points
        if surface_mesh_to_bind is not None:
            ## wait to finish
            self._n_points = len(self.gaussians.get_xyz)
            pass
        else:
            self.binded_to_surface_mesh = False
            n_points = len(self.gaussians.get_xyz)

        # initialize radiues
        # self.scale_activation = scale_activation
        # self.scale_inverse_activation = scale_inverse_activation
        # if not self.binded_to_surface_mesh:
        #     if initialize:
        #         radiuses = _initialize_radiuses_gauss_rasterizer(self, )
        #         print("Initialized radiuses for 3D Gauss Rasterizer")
        #     else:
        #         radiuses = torch.rand(1, n_points, self.radius_dim, device=gaussians.device)
        #         self.min_radius = self.min_ndc_radius / self.focal_factor * 0.005  # 0.005
        #         self.max_radius = self.max_ndc_radius / self.focal_factor * 2.  # 2.
        #
        #     # reset the scaling and rotation for regularization
        #     gaussians.set_scaling(radiuses[0, ..., 4:])
        #     gaussians.set_rotation(radiuses[0, ..., :4])

        self.keep_track_of_knn = keep_track_of_knn
        self.knn_to_track = knn_to_track
        # if keep_track_of_knn:
        # self.knn_to_track = knn_to_track
        # knns = knn_points(gaussians.get_xyz[None], gaussians.get_xyz[None], K=knn_to_track)
        # self.knn_dists = knns.dists[0]
        # self.knn_idx = knns.idx[0]
        # self.reset_neighbors(knn_to_track)

        # Beta mode
        self.beta_mode = beta_mode
        if beta_mode == 'learnable':
            with torch.no_grad():
                log_beta = self.gaussians.get_scaling.mean().log().view(1, )
            self._log_beta = torch.nn.Parameter(
                log_beta.to(self.gaussians.device),
            ).to(self.gaussians.device)

    @property
    def points(self):
        return self.gaussians.get_xyz

    @property
    def scaling(self):
        if not self.binded_to_surface_mesh:
            scales = self.gaussians.get_scaling
        else:
            scales = None
            # scales = torch.cat([
            #     self.surface_mesh_thickness * torch.ones(len(self.gaussians.get_scaling), 1, device=self.device),
            #     self.scale_activation(self._scales)
            # ], dim=-1)
        return scales

    @property
    def strengths(self):
        return self.gaussians.get_opacity

    @property
    def n_points(self):
        if not self.binded_to_surface_mesh:
            return len(self.gaussians.get_xyz)
        else:
            return self._n_points

    @property
    def device(self):
        return self.gaussians.device

    @property
    def quaternions(self):
        if not self.binded_to_surface_mesh:
            quaternions = self.gaussians.get_rotation
        else:
            quaternions = None
        return quaternions

    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                   probabilities_proportional_to_opacity=False,
                                   probabilities_proportional_to_volume=True, ):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if mask is None:
            scaling = self.scaling
        else:
            scaling = self.scaling[mask]

        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0])

        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs()
        cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)

        # sample from multinomial distribution with probability cum_probs, return indices
        # if probabilities_proportional_to_volume and probabilities_proportional_to_opacity are None, then sample uniformly from 0~unmasked points num
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True)

        if mask is not None:
            # get the valid indices for 3d gaussians
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]

        # add noise to point
        random_points = self.points[random_indices] + quaternion_apply(
            self.quaternions[random_indices],
            sampling_scale_factor * self.scaling[random_indices] * torch.randn_like(self.points[random_indices]))

        return random_points, random_indices

    @torch.no_grad()
    def reset_neighbors(self, knn_to_track: int = None):
        if self.binded_to_surface_mesh:
            print("WARNING! You should not reset the neighbors of a surface mesh.")
            print("Then, neighbors reset will be ignored.")
        else:
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
                self.time_knn_dists = None
                self.time_knn_idx = None
                self.ref_timestamps = None


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

    def get_field_values(self, x, gaussian_idx=None,
                         closest_gaussians_idx=None,
                         gaussian_strengths=None,
                         gaussian_centers=None,
                         gaussian_inv_scaled_rotation=None,
                         return_sdf=True, density_threshold=1., density_factor=1.,
                         return_sdf_grad=False, sdf_grad_max_value=10.,
                         opacity_min_clamp=1e-16,
                         return_closest_gaussian_opacities=False,
                         return_beta=False, ):
        if gaussian_strengths is None:
            gaussian_strengths = self.strengths
        if gaussian_centers is None:
            gaussian_centers = self.points
        if gaussian_inv_scaled_rotation is None:
            gaussian_inv_scaled_rotation = self.get_covariance(return_full_matrix=True, return_sqrt=True,
                                                               inverse_scales=True)
        if closest_gaussians_idx is None:
            closest_gaussians_idx = self.knn_idx[gaussian_idx]
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx]
        closest_gaussian_strengths = gaussian_strengths[closest_gaussians_idx]

        fields = {}

        # Compute the density field as a sum of local gaussian opacities
        # TODO: Change the normalization of the density (maybe learn the scaling parameter?)
        shift = (x[:, None] - closest_gaussian_centers)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(
            -1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        fields['density'] = densities.clone()

        # normalize density bigger or equal than 1
        density_mask = densities >= 1.
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)

        if return_closest_gaussian_opacities:
            fields['closest_gaussian_opacities'] = neighbor_opacities

        if return_sdf or return_sdf_grad or return_beta:
            # --- Old way
            # beta = self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
            # --- New way
            beta = self.get_beta(x,
                                 closest_gaussians_idx=closest_gaussians_idx,
                                 closest_gaussians_opacities=neighbor_opacities,
                                 densities=densities,
                                 opacity_min_clamp=opacity_min_clamp,
                                 )
            clamped_densities = densities.clamp(min=opacity_min_clamp)

        if return_beta:
            fields['beta'] = beta

        # Compute the signed distance field
        if return_sdf:
            sdf_values = beta * (
                torch.sqrt(-2. * torch.log(clamped_densities))  # TODO: Change the max=1. to something else?
                - np.sqrt(-2. * np.log(min(density_threshold, 1.)))
            )
            fields['sdf'] = sdf_values

        # Compute the gradient of the signed distance field
        if return_sdf_grad:
            sdf_grad = neighbor_opacities[..., None] * (closest_gaussian_inv_scaled_rotation @ warped_shift)[..., 0]
            sdf_grad = sdf_grad.sum(dim=-2)
            sdf_grad = \
                (beta / (clamped_densities * torch.sqrt(-2. * torch.log(clamped_densities))).clamp(
                    min=opacity_min_clamp))[
                    ..., None] * sdf_grad
            fields['sdf_grad'] = sdf_grad.clamp(min=-sdf_grad_max_value, max=sdf_grad_max_value)

        return fields

    def get_smallest_axis(self, return_idx=False):
        """Returns the smallest axis of the Gaussians.

        Args:
            return_idx (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rotation_matrices = quaternion_to_matrix(self.quaternions)
        # shape (n, 3, 1)
        smallest_axis_idx = self.scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        # get the column value of rotation matrix, the column idx is the smallest axis idx of scaling
        # shape (n, 3, 1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normals(self, estimate_from_points=False, neighborhood_size: int = 32):
        """Returns the normals of the Gaussians.

        Args:
            estimate_from_points (bool, optional): _description_. Defaults to False.
            neighborhood_size (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
        """
        if estimate_from_points:
            normals = estimate_pointcloud_normals(
                self.points[None],  # .detach(),
                neighborhood_size=neighborhood_size,
                disambiguate_directions=True
            )[0]
        else:
            if self.binded_to_surface_mesh:
                ## TODO mesh part wait to be fixed
                normals = torch.nn.functional.normalize(self.surface_mesh.faces_normals_list()[0], dim=-1).view(-1, 1,
                                                                                                                3)
                normals = normals.expand(-1, self.n_gaussians_per_surface_triangle, -1).reshape(-1, 3)
            else:
                normals = self.get_smallest_axis()
        return normals

    def get_beta(self, x,
                 closest_gaussians_idx=None,
                 closest_gaussians_opacities=None,
                 densities=None,
                 opacity_min_clamp=1e-32, ):
        """_summary_

        Args:
            x (_type_): Should have shape (n_points, 3)
            closest_gaussians_idx (_type_, optional): Should have shape (n_points, n_neighbors).
                Defaults to None.
            closest_gaussians_opacities (_type_, optional): Should have shape (n_points, n_neighbors).
            densities (_type_, optional): Should have shape (n_points, ).

        Returns:
            _type_: _description_
        """
        if self.beta_mode == 'learnable':
            return torch.exp(self._log_beta).expand(len(x))

        elif self.beta_mode == 'average':
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='average'.")
            return self.scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)

        elif self.beta_mode == 'weighted_average':
            if closest_gaussians_idx is None:
                raise ValueError("closest_gaussians_idx must be provided when using beta_mode='weighted_average'.")
            if closest_gaussians_opacities is None:
                raise ValueError(
                    "closest_gaussians_opacities must be provided when using beta_mode='weighted_average'.")

            min_scaling = self.scaling.min(dim=-1)[0][closest_gaussians_idx]

            # if densities is None:
            if True:
                opacities_sum = closest_gaussians_opacities.sum(dim=-1, keepdim=True)
            else:
                opacities_sum = densities.view(-1, 1)
            # weights = neighbor_opacities.clamp(min=opacity_min_clamp) / opacities_sum.clamp(min=opacity_min_clamp)
            weights = closest_gaussians_opacities / opacities_sum.clamp(min=opacity_min_clamp)

            # Three methods to handle the case where all opacities are 0.
            # Important because we need to avoid beta == 0 at all cost for these points!
            # Indeed, beta == 0. gives sdf == 0.
            # However these points are far from gaussians, so they should have a sdf != 0.

            # Method 1: Give 1-weight to closest gaussian (Not good)
            if False:
                one_at_closest_gaussian = torch.zeros(1, neighbor_opacities.shape[1], device=rc.device)
                one_at_closest_gaussian[0, 0] = 1.
                weights[opacities_sum[..., 0] == 0.] = one_at_closest_gaussian
                beta = (rc.scaling.min(dim=-1)[0][closest_gaussians_idx] * weights).sum(dim=1)

            # Method 2: Give the maximum scaling value in neighbors as beta (Not good if neighbors have low scaling)
            if False:
                beta = (min_scaling * weights).sum(dim=-1)
                mask = opacities_sum[..., 0] == 0.
                beta[mask] = min_scaling.max(dim=-1)[0][mask]

            # Method 3: Give a constant, large beta value (better control)
            if True:
                beta = (min_scaling * weights).sum(dim=-1)
                with torch.no_grad():
                    if False:
                        # Option 1: beta = camera_spatial_extent
                        beta[opacities_sum[..., 0] == 0.] = rc.get_cameras_spatial_extent()
                    else:
                        # Option 2: beta = largest min_scale in the scene
                        beta[opacities_sum[..., 0] == 0.] = min_scaling.max().detach()

            return beta

        else:
            raise ValueError("Unknown beta_mode.")

    def coarse_density_regulation(self, args):
        # ====================Parameters==================== #
        # num_device = args.gpu
        detect_anomaly = False

        # -----Data parameters-----
        downscale_resolution_factor = 1  # 2, 4
        # -----Model parameters-----
        use_eval_split = True
        n_skip_images_for_eval_split = 8

        freeze_gaussians = False
        initialize_from_trained_3dgs = True  # True or False
        if initialize_from_trained_3dgs:
            prune_at_start = False
            start_pruning_threshold = 0.5
        no_rendering = freeze_gaussians

        n_points_at_start = None  # If None, takes all points in the SfM point cloud

        learnable_positions = True  # True in 3DGS
        use_same_scale_in_all_directions = False  # Should be False
        sh_levels = 4

        # -----Radiance Mesh-----
        triangle_scale = 1.

        # -----Rendering parameters-----
        compute_color_in_rasterizer = False  # TODO: Try True

        # -----Optimization parameters-----

        # Learning rates and scheduling
        num_iterations = 15_000  # Changed

        spatial_lr_scale = None
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 30_000
        feature_lr = 0.0025
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001

        # Densifier and pruning
        heavy_densification = False
        if initialize_from_trained_3dgs:
            densify_from_iter = 500 + 99999  # 500  # Maybe reduce this, since we have a better initialization?
            densify_until_iter = 7000 - 7000  # 7000
        else:
            densify_from_iter = 500  # 500  # Maybe reduce this, since we have a better initialization?
            densify_until_iter = 7000  # 7000

        if heavy_densification:
            densification_interval = 50  # 100
            opacity_reset_interval = 3000  # 3000

            densify_grad_threshold = 0.0001  # 0.0002
            densify_screen_size_threshold = 20
            prune_opacity_threshold = 0.005
            densification_percent_distinction = 0.01
        else:
            densification_interval = 100  # 100
            opacity_reset_interval = 3000  # 3000

            densify_grad_threshold = 0.0002  # 0.0002
            densify_screen_size_threshold = 20
            prune_opacity_threshold = 0.005
            densification_percent_distinction = 0.01

        # Data processing and batching
        n_images_to_use_for_training = -1  # If -1, uses all images

        train_num_images_per_batch = 1  # 1 for full images

        # Loss functions
        loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
        if loss_function == 'l1+dssim':
            dssim_factor = 0.2

        # Regularization
        enforce_entropy_regularization = True
        if enforce_entropy_regularization:
            start_entropy_regularization_from = 7000
            end_entropy_regularization_at = 9000  # TODO: Change
            entropy_regularization_factor = 0.1

        regularize_sdf = True
        if regularize_sdf:
            beta_mode = 'average'  # 'learnable', 'average' or 'weighted_average'

            # start_sdf_regularization_from = 9000
            start_sdf_regularization_from = 7000

            regularize_sdf_only_for_gaussians_with_high_opacity = False
            if regularize_sdf_only_for_gaussians_with_high_opacity:
                sdf_regularization_opacity_threshold = 0.5

            use_sdf_estimation_loss = True
            enforce_samples_to_be_on_surface = False
            if use_sdf_estimation_loss or enforce_samples_to_be_on_surface:
                sdf_estimation_mode = 'density'  # 'sdf' or 'density'
                # sdf_estimation_factor = 0.2  # 0.1 or 0.2?
                samples_on_surface_factor = 0.2  # 0.05

                squared_sdf_estimation_loss = False
                squared_samples_on_surface_loss = False

                normalize_by_sdf_std = False  # False

                # start_sdf_estimation_from = 9000  # 7000
                start_sdf_estimation_from = 7000

                sample_only_in_gaussians_close_to_surface = True
                close_gaussian_threshold = 2.  # 2.

                use_projection_as_estimation = True
                if use_projection_as_estimation:
                    sample_only_in_gaussians_close_to_surface = False

                backpropagate_gradients_through_depth = True  # True

            density_factor = 1. / 16.  # Should be equal to 1. / regularity_knn
            if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and sdf_estimation_mode == 'density':
                density_factor = 1.
            density_threshold = 1.  # 0.5 * density_factor
            sdf_sampling_scale_factor = 1.5
            sdf_sampling_proportional_to_volume = False

        bind_to_surface_mesh = False
        if bind_to_surface_mesh:
            learn_surface_mesh_positions = True
            learn_surface_mesh_opacity = True
            learn_surface_mesh_scales = True
            n_gaussians_per_surface_triangle = 6  # 1, 3, 4 or 6

            use_surface_mesh_laplacian_smoothing_loss = True
            if use_surface_mesh_laplacian_smoothing_loss:
                surface_mesh_laplacian_smoothing_method = "uniform"  # "cotcurv", "cot", "uniform"
                surface_mesh_laplacian_smoothing_factor = 5.  # 0.1

            use_surface_mesh_normal_consistency_loss = True
            if use_surface_mesh_normal_consistency_loss:
                surface_mesh_normal_consistency_factor = 0.1  # 0.1

            densify_from_iter = 999_999
            densify_until_iter = 0
            position_lr_init = 0.00016 * 0.01
            position_lr_final = 0.0000016 * 0.01
            scaling_lr = 0.005
        else:
            surface_mesh_to_bind_path = None

        if regularize_sdf:
            regularize = True
            regularity_knn = 16  # 8 until now
            # regularity_knn = 8
            regularity_samples = -1  # Retry with 1000, 10000
            reset_neighbors_every = 500  # 500 until now
            regularize_from = 7000  # 0 until now
            start_reset_neighbors_from = 7000 + 1  # 0 until now (should be equal to regularize_from + 1?)
            prune_when_starting_regularization = False
        else:
            regularize = False
            regularity_knn = 0
        if bind_to_surface_mesh:
            regularize = False
            regularity_knn = 0

        # Opacity management
        prune_low_opacity_gaussians_at = [9000]
        if bind_to_surface_mesh:
            prune_low_opacity_gaussians_at = [999_999]
        prune_hard_opacity_threshold = 0.5

        # Warmup
        do_resolution_warmup = False
        if do_resolution_warmup:
            resolution_warmup_every = 500
            current_resolution_factor = downscale_resolution_factor * 4.
        else:
            current_resolution_factor = downscale_resolution_factor

        do_sh_warmup = True  # Should be True
        if initialize_from_trained_3dgs:
            do_sh_warmup = False
            sh_levels = 4  # nerfmodel.gaussians.active_sh_degree + 1
        if do_sh_warmup:
            sh_warmup_every = 1000
            current_sh_levels = 1
        else:
            current_sh_levels = sh_levels

        # -----Log and save-----
        print_loss_every_n_iterations = 50
        save_model_every_n_iterations = 1_000_000
        save_milestones = [9000, 12_000, 15_000]

        # new
        # batch_visibility_filter = args.outputs['visibility_filter']
        n_samples_for_sdf_regularization = args.n_samples_for_sdf_regularization
        use_sdf_better_normal_loss = args.use_sdf_better_normal_loss
        sdf_better_normal_gradient_through_normal_only = use_sdf_better_normal_loss
        # ====================Parameters==================== #

        # ====================Regulation loss==================== #
        visibility_filter = torch.ones(self.gaussians._xyz.shape[0]).bool().cuda()
        loss = {"density_regulation": 0, "normal_regulation": 0}

        sampling_mask = visibility_filter
        n_gaussians_in_sampling = sampling_mask.sum()
        if n_gaussians_in_sampling > 0:
            sdf_samples, sdf_gaussian_idx = self.sample_points_in_gaussians(
                num_samples=n_samples_for_sdf_regularization,
                sampling_scale_factor=sdf_sampling_scale_factor,
                mask=sampling_mask,
                probabilities_proportional_to_volume=sdf_sampling_proportional_to_volume,
            )

            fields = self.get_field_values(
                sdf_samples, sdf_gaussian_idx,
                return_sdf=False,
                density_threshold=density_threshold,
                density_factor=density_factor,
                return_sdf_grad=False,
                sdf_grad_max_value=10.,
                return_closest_gaussian_opacities=use_sdf_better_normal_loss,
                return_beta=True)

            # Compute the depth of the points in the gaussians
            proj_mask = torch.ones_like(sdf_samples[..., 0], dtype=torch.bool)
            samples_gaussian_normals = self.get_normals(estimate_from_points=False)[
                sdf_gaussian_idx]
            sdf_estimation = ((sdf_samples - self.points[
                sdf_gaussian_idx]) * samples_gaussian_normals).sum(
                dim=-1)  # Shape is (n_samples,)

            # Compute sdf estimation loss
            beta = fields['beta'][proj_mask]
            densities = fields['density'][proj_mask]
            target_densities = torch.exp(-0.5 * sdf_estimation.pow(2) / beta.pow(2))
            if squared_sdf_estimation_loss:
                sdf_estimation_loss = ((densities - target_densities)).pow(2)
            else:
                sdf_estimation_loss = (densities - target_densities).abs()
            loss["density_regulation"] += sdf_estimation_loss.mean()

            # Compute sdf better normal loss
            if use_sdf_better_normal_loss:
                closest_gaussians_idx = self.knn_idx[sdf_gaussian_idx]
                closest_min_scaling = self.scaling.min(dim=-1)[0][closest_gaussians_idx].detach().view(
                    len(sdf_samples), -1)

                # Compute normals and flip their sign if needed
                gaussians_normals = self.get_normals(estimate_from_points=False)
                closest_gaussian_normals = gaussians_normals[closest_gaussians_idx]
                samples_gaussian_normals = gaussians_normals[sdf_gaussian_idx]
                closest_gaussian_normals = closest_gaussian_normals * torch.sign(
                    (closest_gaussian_normals * samples_gaussian_normals[:, None]).sum(dim=-1,
                                                                                        keepdim=True)).detach()

                # Compute weights for normal regularization, based on the gradient of the sdf
                closest_gaussian_opacities = fields['closest_gaussian_opacities'].detach()
                normal_weights = ((sdf_samples[:, None] - self.points[
                    closest_gaussians_idx]) * closest_gaussian_normals).sum(
                    dim=-1).abs()  # Shape is (n_samples, n_neighbors)
                if sdf_better_normal_gradient_through_normal_only:
                    normal_weights = normal_weights.detach()
                normal_weights = closest_gaussian_opacities * normal_weights / closest_min_scaling.clamp(
                    min=1e-6) ** 2  # Shape is (n_samples, n_neighbors)

                # The weights should have a sum of 1 because of the eikonal constraint
                normal_weights_sum = normal_weights.sum(dim=-1).detach()  # Shape is (n_samples,)
                normal_weights = normal_weights / normal_weights_sum.unsqueeze(-1).clamp(
                    min=1e-6)  # Shape is (n_samples, n_neighbors)

                # Compute regularization loss
                sdf_better_normal_loss = (samples_gaussian_normals - (
                    normal_weights[..., None] * closest_gaussian_normals).sum(dim=-2)
                                            ).pow(2).sum(dim=-1)  # Shape is (n_samples,)
                loss['normal_regulation'] += sdf_better_normal_loss.mean()
        # ====================Regulation loss==================== #
        return loss

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# def getWorld2View(R, t):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = R.transpose()
#     Rt[:3, 3] = t
#     Rt[3, 3] = 1.0
#     return np.float32(Rt)

def getWorld2View(R, t, tensor=False):
    if tensor:
        Rt = torch.zeros(4, 4, device=R.device)
        Rt[..., :3, :3] = R.transpose(-1, -2)
        Rt[..., :3, 3] = t
        Rt[..., 3, 3] = 1.0
        return Rt
    else:
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))