import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import numpy as np
from models import tsg_centroid_module, tsg_seg_module
from models.tsg_loss import centroid_loss, segmentation_loss, segmentation_mask_loss
from torch.optim.lr_scheduler import ExponentialLR
import models.gen_utils as gen_utils
import models.tsg_utils as utils
from models.generator import CenterPointGenerator

MEMORY_TEST = True

centroid_model = tsg_centroid_module.get_model()
centroid_model.load_state_dict(torch.load("pretrained_cent_model.h5"))
centroid_model.cuda()

SAMPLING_NUM = 2048

point_loader = DataLoader(CenterPointGenerator("data/sampled_val"), batch_size=1)
for epoch in range(10000):
    for batch_item in point_loader:
        points = batch_item[0].cuda()
        centroid_coords = batch_item[1].cuda()
        centroid_labels = batch_item[3].cuda()#1, 1, num of centroids
        seg_label = batch_item[2].cuda()

        with torch.no_grad():
            y_center_pred = centroid_model(points)
            centroid_network_loss = centroid_loss(y_center_pred[4], y_center_pred[3], y_center_pred[5], centroid_coords)
        
        output_offset_cpu = y_center_pred[4].cpu().detach().numpy()[0,:].T
        output_xyz_cpu = y_center_pred[3].cpu().detach().numpy()[0,:].T
        cen_cpu = centroid_coords.cpu().detach().numpy()[0,:].T
        output_dist_cpu = y_center_pred[5].cpu().detach().numpy()[0,:].T
        output_input_xyz_cpu = y_center_pred[2].cpu().detach().numpy()[0,:].T
        
        sampled_db_scan = utils.dbscan_pc(y_center_pred[3], y_center_pred[4], y_center_pred[5])
        nearest_n = utils.get_nearest_neighbor_idx(y_center_pred[2], sampled_db_scan, SAMPLING_NUM)
        cropped_coords, _ = utils.get_nearest_neighbor_points_with_centroids(y_center_pred[2], nearest_n, sampled_db_scan)
        cropped_gt_labels, sampled_db_scan_cuda = utils.get_nearest_neighbor_points_with_centroids(seg_label, nearest_n, sampled_db_scan)

        o3d.visualization.draw_geometries([gen_utils.np_to_pcd(output_xyz_cpu, color=[0,1,0]), gen_utils.np_to_pcd(cen_cpu)], mesh_show_back_face=False,mesh_show_wireframe=True)
        o3d.visualization.draw_geometries([gen_utils.np_to_pcd(output_offset_cpu+output_xyz_cpu, color=[0,1,0]), gen_utils.np_to_pcd(cen_cpu)], mesh_show_back_face=False,mesh_show_wireframe=True)
        o3d.visualization.draw_geometries([gen_utils.np_to_pcd(sampled_db_scan[0], color=[0,1,0]), gen_utils.np_to_pcd(cen_cpu)], mesh_show_back_face=False,mesh_show_wireframe=True)
        centroid_coords = centroid_coords.repeat(len(sampled_db_scan_cuda), 1, 1)
        centroid_labels = centroid_labels.repeat(len(sampled_db_scan_cuda), 1, 1)
        #gen_utils.crop_visualization(cropped_coords[:,:3], cropped_gt_labels, sampled_db_scan_cuda, centroid_coords, centroid_labels)
