import torch
from torch.utils.data import Dataset
import os
import numpy as np
from glob import glob
class CenterPointGenerator(Dataset):
    def __init__(self, data_dir="data/sampled_cls_full"):
        self.data_dir = data_dir
        self.centroid_coords_paths = glob(os.path.join(data_dir,"*_centroid.npy"))
        self.centroid_exists_paths = glob(os.path.join(data_dir,"*_centroid_exist.npy"))
        self.mesh_paths = glob(os.path.join(data_dir,"*_mesh.npy"))
    
    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        #low = o3d.io.read_point_cloud(os.path.join("data", "case1", "sampled", "align_low.ply"))

        #low_arr = np.asarray(low.points).astype("float32")
        #low_n = np.asarray(low.normals).astype("float32")
        #low_feat = np.concatenate((low_arr,low_n), axis=1)
        mesh_arr = np.load(self.mesh_paths[idx])

        low_feat = mesh_arr.copy()[:,:6].astype("float32")
        low_feat = torch.from_numpy(low_feat)
        low_feat = low_feat.permute(1,0)
        
        #B,16
        centroid_arr = np.load(self.centroid_coords_paths[idx])
        centroid_coords = centroid_arr.copy()[:,:3].astype("float32")
        centroid_coords = torch.from_numpy(centroid_coords)
        centroid_coords = centroid_coords.permute(1,0)

        seg_label = mesh_arr.copy()[:,6:].astype("int")
        
        # 1번 치아는 모델에서 0번
        seg_label -= 1
        seg_label = torch.from_numpy(seg_label)
        seg_label = seg_label.permute(1,0)

        #B,16
        centroid_exists = np.load(self.centroid_exists_paths[idx]).astype(int)
        centroid_exists = torch.from_numpy(centroid_exists)

        return low_feat, centroid_coords, seg_label, centroid_exists