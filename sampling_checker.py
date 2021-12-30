import torch
import sys
import importlib
import models.tsg_loss as tsg_loss
import models
from models import tsg_centroid_module, tsg_seg_module
from models.generator import CenterPointGenerator
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import models.tsg_utils as utils
import models.gen_utils as gen_utils

CROP_SAMPLING_NUM = 2048
point_loader = DataLoader(CenterPointGenerator(), batch_size=1, shuffle=False)

for batch_item in point_loader:
    visualization_arr = np.concatenate([batch_item[0][0].T, batch_item[2][0].T], axis=1)
    centroids = batch_item[1][0].T
    gen_utils.print_3d([gen_utils.np_to_by_label(visualization_arr, axis=6), gen_utils.np_to_pcd(centroids[:,:3])])
    
    
    points = batch_item[0].cuda()
    centroid_coords = batch_item[1].cuda()
    centroid_labels = batch_item[3].cuda()#1, 1, num of centroids
    seg_label = batch_item[2].cuda()

    sampled_db_scan = [centroid_coords[0].cpu().detach().numpy().T]
    nearest_n = utils.get_nearest_neighbor_idx(points[:,:3], sampled_db_scan, CROP_SAMPLING_NUM)
    cropped_coords, _ = utils.get_nearest_neighbor_points_with_centroids(points, nearest_n, sampled_db_scan)
    cropped_gt_labels, sampled_db_scan_cuda = utils.get_nearest_neighbor_points_with_centroids(seg_label, nearest_n, sampled_db_scan)
    centroid_coords = centroid_coords.repeat(len(sampled_db_scan_cuda), 1, 1)
    centroid_labels = centroid_labels.repeat(len(sampled_db_scan_cuda), 1, 1)
    gen_utils.crop_visualization(cropped_coords[:,:3], cropped_gt_labels, sampled_db_scan_cuda, centroid_coords, centroid_labels)
