import torch
import sys
import os
sys.path.append(os.getcwd())
import importlib
import models.tsg_loss as tsg_loss
import models
from models import tsg_centroid_classify_module
from models.generator_classify import CenterPointGenerator
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import models.tsg_utils as utils
import models.gen_utils as gen_utils

cuda = torch.device('cuda')

centroid_model = tsg_centroid_classify_module.get_model()
centroid_model.load_state_dict(torch.load("ckpt_cls/0102_cent_train_full_0103.h5"))
centroid_model.cuda()
CROP_SAMPLING_NUM = 2048
point_loader = DataLoader(CenterPointGenerator(), batch_size=1, shuffle=False)


def move_sampled_point(pred_offset, sampled_coord, gt_cent_coord):
    # pred_offset : B, 24, encoder_sampled_points
    # sample_coord : B, 3, encoder_sampled_points
    # gt_centroid_coord : B, 3, 16(2*8)
    # gt_cent_exist : B,8

    B, _, S_N = sampled_coord.shape
    gt_cent_coord = gt_cent_coord.permute(0,2,1)
    sampled_coord = sampled_coord.permute(0,2,1)
    sampled_coord = sampled_coord.view(B, 1, S_N, 3)
    pred_offset = pred_offset.view(B, 8, 3, S_N)
    pred_offset = pred_offset.permute(0,1,3,2)
    #B, 8, encdoer_sampeld_points, 3
    moved_coord = sampled_coord + pred_offset
    return moved_coord

for batch_item in point_loader:
    points = batch_item[0].cuda()
    gt_centroid_coords = batch_item[1].cuda()
    gt_centroid_exists = batch_item[3].cuda()
    with torch.no_grad():
        y_center_pred = centroid_model(points)

    l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, exist_pred =  y_center_pred
    moved_point = move_sampled_point(offset_result, l3_xyz, gt_centroid_coords)
    #moved_point = l3_xyz
    sampled_points = l3_xyz.cpu().detach().numpy()[0,:].T
    for i in range(8):
        pred_dist_cpu = dist_result[0,i,:].cpu().detach().numpy()
        moved_points_cpu = moved_point[0,i,:,:].cpu().detach().numpy()
        #moved_points_cpu = moved_point[0,:,:].cpu().detach().numpy().T
        moved_points_cpu+= 0.01
        moved_points_cpu = moved_points_cpu[pred_dist_cpu<0.03]
        gen_utils.print_3d([gen_utils.np_to_pcd(sampled_points, color=[0,1,0]), gen_utils.np_to_pcd(moved_points_cpu)])

    """
    visualization_arr = np.concatenate([batch_item[0][0].T, batch_item[2][0].T], axis=1)

    centroids = batch_item[1][0].T
    centroid_mark_arr = np.array([np.arange(8),np.arange(8)]).T.reshape(16,1)+1
    centroids = np.concatenate([centroids, centroid_mark_arr], axis=1)
    centroids = centroids[centroids[:,0]>-5]
    gen_utils.print_3d([gen_utils.np_to_by_label(visualization_arr, axis=6), gen_utils.np_to_by_label(centroids)])
    
    
    points = batch_item[0].cuda()
    centroid_coords = batch_item[1].cuda()
    centroid_exists= batch_item[3].cuda()#1, 1, num of centroids
    seg_label = batch_item[2].cuda()
    print(centroid_exists)
    sampled_db_scan = [centroid_coords[0].cpu().detach().numpy().T]
    nearest_n = utils.get_nearest_neighbor_idx(points[:,:3], sampled_db_scan, CROP_SAMPLING_NUM)
    cropped_coords, _ = utils.get_nearest_neighbor_points_with_centroids(points, nearest_n, sampled_db_scan)
    cropped_gt_labels, sampled_db_scan_cuda = utils.get_nearest_neighbor_points_with_centroids(seg_label, nearest_n, sampled_db_scan)
    centroid_coords = centroid_coords.repeat(len(sampled_db_scan_cuda), 1, 1)
    #centroid_labels = centroid_labels.repeat(len(sampled_db_scan_cuda), 1, 1)
    #gen_utils.crop_visualization(cropped_coords[:,:3], cropped_gt_labels, sampled_db_scan_cuda, centroid_coords, centroid_labels)
    """