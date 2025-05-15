import os
import torch
import numpy as np
import torch.utils.data as data
import random
from .build import DATASETS
import open3d as o3d
import open3d
import trimesh
from os import listdir
import logging
import copy
from models.PoinTr import fps
from SAP.src.dpsr import DPSR
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@DATASETS.register_module()
class crown(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        self.dpsr = DPSR(res=(128, 128, 128), sig = 2)

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            tax_id = line
            if 'Lower' in tax_id:
                taxonomy_id = '0'
            else:
                taxonomy_id = '1'
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': tax_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        std_pc = np.std(pc, axis=0)
        pc = (pc - centroid) / std_pc
        return pc, centroid, std_pc


    def normalize_points_mean_std(self, main, opposing, shell):

        # new_context = copy.deepcopy(main)
        # new_opposing = copy.deepcopy(opposing)
        # new_crown = copy.deepcopy(shell)
        # new_marginline = copy.deepcopy(marginline)

        context_mean, context_std = np.mean(np.concatenate((main.points, opposing.points), axis=0), axis=0), \
                                    np.std(np.concatenate((main.points, opposing.points), axis=0), axis=0)
        # scale values
        new_context_points = (np.asarray(main.points) - context_mean) / context_std
        # new_context.points = o3d.utility.Vector3dVector(new_context_points)

        # final_context = copy.deepcopy(new_context)

        new_opposing_points = (np.asarray(opposing.points) - context_mean) / context_std
        # new_opposing.points = o3d.utility.Vector3dVector(new_opposing_points)

        new_crown_points = (np.asarray(shell.points) - context_mean) / context_std
        # new_crown.points = o3d.utility.Vector3dVector(new_crown_points)
        # new_marginline_points = (np.asarray(marginline.points) - context_mean) / context_std

        return new_context_points, new_opposing_points, new_crown_points, context_mean, context_std

    def normalize_points_dental(self, main, opposing, shell=None):
    
        # Compute joint mean and max distance based on main and opposing only
        combined_points = np.concatenate((main.points, opposing.points), axis=0)
        context_mean = np.mean(combined_points, axis=0)
        centered_combined = combined_points - context_mean
        max_distance_combined = np.max(np.linalg.norm(centered_combined, axis=1))
        
        # Center and scale main and opposing
        new_context_points = (np.asarray(main.points) - context_mean) / max_distance_combined
        new_opposing_points = (np.asarray(opposing.points) - context_mean) / max_distance_combined
        
        # Center and scale shell (if provided, e.g., during training)
        new_crown_points = None
        if shell is not None:
            new_crown_points = (np.asarray(shell.points) - context_mean) / max_distance_combined
        
        return new_context_points, new_opposing_points, new_crown_points, context_mean, max_distance_combined


    def __getitem__(self, idx):

        # read points
        sample = self.file_list[idx]
        # print(sample['file_path'])
        
        for j in os.listdir(os.path.join(self.pc_path, sample['file_path'])):
            if 'antagonist' in j:
                opposing = o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))

                
            if 'master' in j:
                main = o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))
                
                

            if 'shell' in j:
                # shell = o3d.io.read_point_cloud(os.path.join(self.pc_path, sample['file_path'], j))
                mesh_tmp = trimesh.load_mesh(os.path.join(self.pc_path, sample['file_path'], j))
                verts = torch.from_numpy(mesh_tmp.vertices[None]).float()
                faces = torch.from_numpy(mesh_tmp.faces[None])
                mesh = Meshes(verts=verts, faces=faces)

                points, normals = sample_points_from_meshes(mesh, num_samples=1536, return_normals=True)

                # 创建点云
                shell = o3d.geometry.PointCloud()
                shell_points = points.squeeze(0).detach().numpy()
                shell.points = o3d.utility.Vector3dVector(shell_points)

                # 保存shell
                # o3d.io.write_point_cloud('shell.ply', shell)

                shellP = np.asarray(shell.points)
                shell_min = np.min(shellP)
                shell_max = np.max(shellP)

                # 创建points的拷贝用于DPSR
                points_dpsr = points.clone()  # 创建一个新的tensor
    
                # 3. 计算归一化参数
                center = verts.mean(1, keepdim=True)  # [1, 1, 3] - 修改维度以匹配points
                vertices_np = verts.numpy()
                center_np = center.numpy()
                scale = np.max(np.max(np.abs(vertices_np - center_np), axis=1))
                scale = torch.tensor(scale, device=points.device, dtype=points.dtype)
    
                # 4. 对points_dpsr进行归一化处理
                points_dpsr = points_dpsr - center  # center现在是[1, 1, 3]，可以正确广播
                points_dpsr = points_dpsr / scale
                points_dpsr  *= 0.9
                # make sure the points are within the range of [0, 1)
                points_dpsr  = points_dpsr  / 2. + 0.5

                shell_grid = self.dpsr(points_dpsr, normals).squeeze(0)

        # normalizie
        try:
            main_only, opposing_only, shell = copy.deepcopy(main), copy.deepcopy(opposing), copy.deepcopy(shell)
        except:
            print(sample['file_path'])
        
        main_only, opposing_only, shell, centroid, std_pc = self.normalize_points_dental(main_only, opposing_only, shell)
        
        antag_pc = torch.from_numpy(opposing_only).float().unsqueeze(0)
        # Set PyTorch random seed for deterministic FPS
        # torch.manual_seed(42)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(42)
        antag_sample = fps(antag_pc, 5120, device)
        master_pc = torch.from_numpy(main_only).float().unsqueeze(0)
        master_sample = fps(master_pc, 5120, device)
        data_partial = torch.concat((master_sample.squeeze(0), antag_sample.squeeze(0)))

        # data_partial_all = torch.concat((master_pc.squeeze(0), antag_pc.squeeze(0)))
        # # 保存data_partial
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data_partial_all.cpu().numpy())
        # o3d.io.write_point_cloud('data_partial_all.ply', pcd)

        # # 保存shell
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(shell)
        # o3d.io.write_point_cloud('gt.ply', pcd)

        # # 保存data_partial
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data_partial.cpu().numpy())
        # o3d.io.write_point_cloud('data_partial.ply', pcd)
        
        data_gt = torch.from_numpy(shell).float()
        min_gt = torch.from_numpy(np.asarray(shell_min)).float()
        max_gt = torch.from_numpy(np.asarray(shell_max)).float()

        value_centroid = torch.from_numpy(centroid).float()
        # value_std_pc = torch.from_numpy(std_pc).float()
        value_std_pc = np.array(std_pc)  # 确保是数组  max_distance_combined
        value_std_pc = torch.from_numpy(value_std_pc).float()

        shell_grid_gt = torch.from_numpy(np.asarray(shell_grid)).float()

        return sample['taxonomy_id'], sample['model_id'], data_gt, data_partial, value_centroid, value_std_pc, shell_grid_gt, min_gt, max_gt

    def __len__(self):
        return len(self.file_list)
