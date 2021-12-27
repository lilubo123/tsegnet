import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import numpy as np
from models import tsg_centroid_module, tsg_seg_module
from models.tsg_loss import centroid_loss
from torch.optim.lr_scheduler import ExponentialLR
import models.tsg_utils as utils


class CenterPointGenerator(Dataset):
    def __init__(self, data_dir="data/case1/sampled"):
        self.data_dir = data_dir

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #low = o3d.io.read_point_cloud(os.path.join("data", "case1", "sampled", "align_low.ply"))

        #low_arr = np.asarray(low.points).astype("float32")
        #low_n = np.asarray(low.normals).astype("float32")
        #low_feat = np.concatenate((low_arr,low_n), axis=1)

        low_feat = np.load(os.path.join("data","case1","sampled","align_low_sampled.npy"))[:,:6].astype("float32")
        low_feat = torch.from_numpy(low_feat)
        low_feat = low_feat.permute(1,0)
        
        centroid = np.load(os.path.join("data","case1","sampled","align_low.txt.npy")).astype("float32")
        centroid = torch.from_numpy(centroid)
        centroid = centroid.permute(1,0)

        seg_label = np.load(os.path.join("data","case1","sampled","align_low_sampled.npy"))[:,6:].astype("int")
        seg_label = torch.from_numpy(seg_label)
        seg_label = seg_label.permute(1,0)
        return low_feat, centroid, seg_label

cuda = torch.device('cuda')

centroid_model = tsg_centroid_module.get_model()
centroid_model.cuda()
centroid_model.load_state_dict(torch.load("pretrained_centroid_model.h5"))
centroid_model.eval()

seg_model = tsg_seg_module.get_model()
seg_model.cuda()

optimizer = torch.optim.Adam(centroid_model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, 0.999)
point_loader = DataLoader(CenterPointGenerator(), batch_size=1)
for epoch in range(10000):
    for batch_item in point_loader:
        points = batch_item[0].cuda()
        centroids = batch_item[1].cuda()
        with torch.no_grad():
            y_center_pred = centroid_model(points)
        #return x, l3_points, l0_xyz, l3_xyz, offset_result, dist_result
        #def centroid_loss(pred_offset, sample_xyz, distance, centroid):
        #loss = centroid_loss(y_center_pred[4], y_center_pred[3], y_center_pred[5], centroids)
        
        sampled_db_scan = utils.dbscan_pc(y_center_pred[3], y_center_pred[4], y_center_pred[5])
        nearest_n = utils.get_nearest_neighbor_idx(y_center_pred[2], sampled_db_scan)
        cropped_coords, _ = utils.get_nearest_neighbor_points_with_centroids(y_center_pred[2], nearest_n, sampled_db_scan)
        cropped_features, sampled_centroids = utils.get_nearest_neighbor_points_with_centroids(y_center_pred[0], nearest_n, sampled_db_scan)
        seg_input = utils.concat_seg_input(cropped_features, cropped_coords, sampled_centroids)

        print(seg_input.shape)
        optimizer.zero_grad()
        batch_size = 8
        for batch_start_idx in range(0, seg_input.shape[0], batch_size):
            pred_seg = seg_model(seg_input[batch_start_idx:batch_start_idx+batch_size, :, :])
            temp_loss = torch.sum(pred_seg[0])
            temp_loss.backward()
        #loss.backward()
        optimizer.step()
        scheduler.step()
        #torch.save(model.state_dict(), "model_2")