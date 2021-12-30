import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import numpy as np
from models import tsg_centroid_module, tsg_seg_module
from models.tsg_loss import centroid_loss, segmentation_loss, segmentation_mask_loss
from torch.optim.lr_scheduler import ExponentialLR
import models.tsg_utils as utils
import models.gen_utils as gen_utils
from models.generator import CenterPointGenerator

MEMORY_TEST = False

centroid_model = tsg_centroid_module.get_model()
centroid_model.cuda()
centroid_model.load_state_dict(torch.load("pretrained_cent_model.h5"))

SAMPLING_NUM = 2048

seg_model = tsg_seg_module.get_model()
seg_model.cuda()

optimizer = torch.optim.Adam(list(centroid_model.parameters()) + list(seg_model.parameters()), lr=1e-4)
scheduler = ExponentialLR(optimizer, 0.999)
point_loader = DataLoader(CenterPointGenerator(), batch_size=1)
val_point_loader = DataLoader(CenterPointGenerator("data/sampled_val"), batch_size=1)

best_val_loss = -100
for epoch in range(10000):
    total_loss = 0
    centroid_model.train()
    seg_model.train()
    for batch_item in point_loader:
        points = batch_item[0].cuda()
        centroid_coords = batch_item[1].cuda()
        centroid_labels = batch_item[3].cuda()#1, 1, num of centroids
        seg_label = batch_item[2].cuda()

        center_model_output = centroid_model(points)

        centroid_network_loss = centroid_loss(center_model_output[4], center_model_output[3], center_model_output[5], centroid_coords)
        
        if MEMORY_TEST:
            print("on memory test")
            sampled_db_scan = [centroid_coords[0].cpu().detach().numpy().T]
        else:
            sampled_db_scan = utils.dbscan_pc(center_model_output[3], center_model_output[4], center_model_output[5])
        # sampled_db_scan : B, 3, pred cluster centroid num
        nearest_n = utils.get_nearest_neighbor_idx(center_model_output[2], sampled_db_scan, SAMPLING_NUM)
        try:
            rand_cluster_indexes = [np.random.randint(0,len(nearest_n[i])) for i in range(len(nearest_n))]
            
            cropped_coords, _ = utils.get_nearest_neighbor_points_hold_batch(center_model_output[2], nearest_n, sampled_db_scan, rand_cluster_indexes)
            cropped_gt_labels, _ = utils.get_nearest_neighbor_points_hold_batch(seg_label, nearest_n, sampled_db_scan, rand_cluster_indexes)
            cropped_features, sampled_centroids = utils.get_nearest_neighbor_points_hold_batch(center_model_output[0], nearest_n, sampled_db_scan, rand_cluster_indexes)
            seg_input = utils.concat_seg_input(cropped_features, cropped_coords, sampled_centroids)
        except Exception as e:
            print(e)
            centroid_network_loss.backward()
            continue
        seg_model_output = seg_model(seg_input)

        pred_cluster_gt_ids = utils.get_cluster_gt_id_by_nearest_gt_centroid(centroid_coords, centroid_labels, sampled_centroids)
        pred_cluster_gt_points_bin_labels = utils.get_cluster_points_bin_label(cropped_gt_labels, pred_cluster_gt_ids)
        #gen_utils.crop_visualization(cropped_coords[:,:3], cropped_gt_labels, sampled_centroids, centroid_coords, centroid_labels)

        seg_network_loss = segmentation_loss(seg_model_output[0], seg_model_output[1], seg_model_output[2], seg_model_output[3], pred_cluster_gt_ids, pred_cluster_gt_points_bin_labels)
        


        print("centroid_network_loss", centroid_network_loss)
        print("segmentation_network_loss", seg_network_loss)

        loss = centroid_network_loss + seg_network_loss
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print("train_loss", total_loss)
    torch.save(seg_model.state_dict(), "model_segmentation_recent_train")
    torch.save(centroid_model.state_dict(), "model_centroid_recent_train")


    total_val_loss = 0
    centroid_model.eval()
    seg_model.eval()
    for batch_item in val_point_loader:
        points = batch_item[0].cuda()
        centroid_coords = batch_item[1].cuda()
        centroid_labels = batch_item[3].cuda()#1, 1, num of centroids
        seg_label = batch_item[2].cuda()
        
        center_model_output = centroid_model(points)
        with torch.no_grad():
            centroid_network_loss = centroid_loss(center_model_output[4], center_model_output[3], center_model_output[5], centroid_coords)
        
        if MEMORY_TEST:
            print("on memory test")
            sampled_db_scan = [centroid_coords[0].cpu().detach().numpy().T]
        else:
            sampled_db_scan = utils.dbscan_pc(center_model_output[3], center_model_output[4], center_model_output[5])
        # sampled_db_scan : B, 3, pred cluster centroid num
        nearest_n = utils.get_nearest_neighbor_idx(center_model_output[2], sampled_db_scan, SAMPLING_NUM)
        try:
            rand_cluster_indexes = [np.random.randint(0,len(nearest_n[i])) for i in range(len(nearest_n))]
            
            cropped_coords, _ = utils.get_nearest_neighbor_points_hold_batch(center_model_output[2], nearest_n, sampled_db_scan, rand_cluster_indexes)
            cropped_gt_labels, _ = utils.get_nearest_neighbor_points_hold_batch(seg_label, nearest_n, sampled_db_scan, rand_cluster_indexes)
            cropped_features, sampled_centroids = utils.get_nearest_neighbor_points_hold_batch(center_model_output[0], nearest_n, sampled_db_scan, rand_cluster_indexes)
            seg_input = utils.concat_seg_input(cropped_features, cropped_coords, sampled_centroids)
        except Exception as e:
            print("in val,,")
            print(e)
            continue
        with torch.no_grad():
            seg_model_output = seg_model(seg_input)

        pred_cluster_gt_ids = utils.get_cluster_gt_id_by_nearest_gt_centroid(centroid_coords, centroid_labels, sampled_centroids)
        pred_cluster_gt_points_bin_labels = utils.get_cluster_points_bin_label(cropped_gt_labels, pred_cluster_gt_ids)
        #gen_utils.crop_visualization(cropped_coords[:,:3], cropped_gt_labels, sampled_centroids, centroid_coords, centroid_labels)

        seg_network_loss = segmentation_loss(seg_model_output[0], seg_model_output[1], seg_model_output[2], seg_model_output[3], pred_cluster_gt_ids, pred_cluster_gt_points_bin_labels)
        
        print("centroid_network_loss", centroid_network_loss)
        print("segmentation_network_loss", seg_network_loss)

        loss = centroid_network_loss + seg_network_loss
        total_val_loss += loss
    print("total_val_loss", total_val_loss)
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(seg_model.state_dict(), "model_segmentation_val_")
        torch.save(centroid_model.state_dict(), "model_centroid_val_")