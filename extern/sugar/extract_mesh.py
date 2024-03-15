import os
import json
import argparse
import numpy as np
from sugar_utils.general_utils import str2bool
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar

import sys
sys.path.append(".")

from threestudio.data.uncond import RandomCameraIterableDataset, RandomCameraDataModuleConfig
from threestudio.utils.config import load_config, parse_structured
from threestudio.utils.ops import convert_pose

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to extract a mesh from a coarse SuGaR scene.')
    parser.add_argument('-s', '--scene_path',
                        type=str, 
                        default="./load/scene",
                        help='path to the scene data to use.')
    parser.add_argument('-c', '--checkpoint_path', 
                        type=str, 
                        help='path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load', 
                        type=int, default=7000, 
                        help='iteration to load.')
    
    parser.add_argument('-m', '--coarse_model_path', type=str, default=None, help='')
    
    parser.add_argument('-l', '--surface_level', type=float, default=0.3, 
                        help='Surface level to extract the mesh at. If None, will extract levels 0.1, 0.3 and 0.5')
    parser.add_argument('-d', '--decimation_target', type=int, default=200_000, 
                        help='Target number of vertices to decimate the mesh to. If None, will decimate to 200_000 and 1_000_000.')
    
    parser.add_argument('-o', '--mesh_output_dir',
                        type=str, default=None, 
                        help='path to the output directory.')
    
    parser.add_argument('-b', '--bboxmin', type=str, default=None, help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None, help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=False, help='If True, center the bounding box. Default is False.')
    
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    
    parser.add_argument('--eval', type=str2bool, default=False, help='Use eval split.')
    parser.add_argument('--use_centers_to_extract_mesh', type=str2bool, default=False, 
                        help='If True, just use centers of the gaussians to extract mesh.')
    parser.add_argument('--use_marching_cubes', type=str2bool, default=False, 
                        help='If True, use marching cubes to extract mesh.')
    parser.add_argument('--use_vanilla_3dgs', action="store_true", default=False, 
                        help='If True, use vanilla 3DGS to extract mesh.')
    parser.add_argument("--camera_pose_path", type=str, default=None,
                        help='The path of .json file that stores camera poses.')
    
    parser.add_argument("--poisson_depth", type=int, default=6,
                        help="The depth used for poisson reconstruction")
    
    args = parser.parse_args()

    if args.camera_pose_path is None:
        gs_path = args.checkpoint_path
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(gs_path)), "configs", "parsed.yaml"
        )
        cfg = load_config(cfg_path)

        random_camera_cfg = parse_structured(
            RandomCameraDataModuleConfig, cfg.get('data', {})
        )
        camera_distance = random_camera_cfg.eval_camera_distance
        random_camera_cfg.update(
            {
                "camera_distance_range": [1.3, 2.0],
                "elevation_range": [-45, 80],
                "azimuth_range": [-180, 180]
            }
        )
        random_camera_dataset = RandomCameraIterableDataset(random_camera_cfg)

        cfg_list = []
        for i in range(1000):
            random_camera = random_camera_dataset.collate(None)
            # convert threestudio pose to 3d gaussian
            c2w = convert_pose(random_camera['c2w'][0].cpu()).numpy()
            camera_cfg = {}
            camera_cfg['id'] = i
            camera_cfg['img_name'] = f"{i}"
            camera_cfg['width'] = 512
            camera_cfg['height'] = 512
            camera_cfg['position'] = c2w[:3, 3].tolist()
            rot = c2w[:3, :3]
            camera_cfg['rotation'] = [x.tolist() for x in rot]

            fov = random_camera_cfg.eval_fovy_deg / 180 * np.pi
            focal_length = 0.5 * 512 / np.tan(0.5 * fov)
            camera_cfg['fx'] = focal_length
            camera_cfg['fy'] = focal_length
            cfg_list.append(camera_cfg)

        camera_save_path = os.path.join(
            os.path.dirname(gs_path), "cameras.json"
        )
        with open(camera_save_path, 'w') as f:
            json.dump(cfg_list, f, indent=4)

    
    # Call function
    extract_mesh_from_coarse_sugar(args)
    