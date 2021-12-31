import open3d as o3d
import numpy as np
import torch
import os
import models.tsg_utils as tsg_utils

def np_to_pcd(arr, color=[1,0,0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def np_to_by_label(arr, axis=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2]]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd
    
def get_gt_mask_labels(pred_mask_1, pred_weight_1, gt_label):
    # pred_mask_1: batch_size, 2, cropped
    # pred_weight_1: batch_size, 1, cropped
    # gt_label: batch_size, 1, cropped

    gt_label = gt_label.view(gt_label.shape[0],-1)
    gt_bin_label = torch.ones_like(gt_label)
    gt_bin_label[gt_label == -1] = 0
    gt_bin_label = gt_bin_label.type(torch.long)
    
    return gt_bin_label

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def get_number_from_name(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def get_up_from_name(path):
    return os.path.basename(path).split("_")[-1].split(".")[0]=="up"

def np_to_by_label(arr, axis=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2]]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def cropped_to_global_label_marker(org_points, cropped_tooth_exists, cropped_weights, cropped_tooth_num, nearest_indexes, binary_output=False):
    # org_points : 3, 16000
    # cropped_tooth_exists : num of cluster, 1 , num of points inside cluster
    # cropped_weights : num of cluster, 1 , num of points inside cluster
    # cropped_tooth_num : num of cluster, 8
    # nearest_indexes : num of cluster, num of points inside cluster

    labeled_points = org_points.cpu().detach().numpy().T
    cropped_tooth_num = cropped_tooth_num.cpu().detach().numpy()
    cropped_tooth_exists = cropped_tooth_exists.view(cropped_tooth_exists.shape[0], cropped_tooth_exists.shape[2])
    cropped_weights = cropped_weights.view(cropped_weights.shape[0], cropped_weights.shape[2])
    labeled_points = np.concatenate([labeled_points, np.zeros((labeled_points.shape[0],1))], axis=1)
    for cluster_idx in range(len(nearest_indexes)):
        labeled_points[nearest_indexes[cluster_idx]]

        cropped_tooth_exists_inside_cluster = cropped_tooth_exists[cluster_idx, :].cpu().detach().numpy()
        cropped_tooth_weights_inside_cluster = cropped_weights[cluster_idx, :].cpu().detach().numpy()
        cropped_tooth_exists_inside_cluster = sigmoid(cropped_tooth_exists_inside_cluster)
        cropped_tooth_weights_inside_cluster = sigmoid(cropped_tooth_weights_inside_cluster)
        cropped_indexes_have_label_arr = nearest_indexes[cluster_idx][(cropped_tooth_exists_inside_cluster)>=0.5]
        if binary_output:
            labeled_points[cropped_indexes_have_label_arr, 3] = 1
        else:
            labeled_points[cropped_indexes_have_label_arr, 3] = np.argmax(cropped_tooth_num[cluster_idx])+1
    return labeled_points

def crop_visualization(cropped_coords, cropped_gt_labels, pred_centroid_coords, gt_centorids_coords, gt_centroids_ids):
    #cropped_gt_labels: num of clsuters (B), features of points, num of points inside cluster (cuda)
    #cropped_coords: num of clsuters (B), 3, num of points inside cluster (cuda)
    #sampled_centroids: num of clusters, 3, 1
    pred_cluster_gt_ids = tsg_utils.get_cluster_gt_id_by_nearest_gt_centroid(gt_centorids_coords, gt_centroids_ids, pred_centroid_coords)
    gt_bin = tsg_utils.get_cluster_points_bin_label(cropped_gt_labels, pred_cluster_gt_ids)
    gt_bin = gt_bin.cpu().detach().numpy()
    cropped_coords_cpu = cropped_coords.cpu().detach().numpy()
    for i in range(cropped_gt_labels.shape[0]):
        temp_gt_bin = gt_bin[i,:].reshape(-1,1)
        temp_coord = cropped_coords_cpu[i, :3, :]
        coords_with_label = np.concatenate([temp_coord.T, temp_gt_bin], axis=1)
        o3d.visualization.draw_geometries([np_to_by_label(coords_with_label, axis=3)], mesh_show_back_face=False,mesh_show_wireframe=True)

def print_3d(data_3d_ls):
    o3d.visualization.draw_geometries(data_3d_ls, mesh_show_back_face=False,mesh_show_wireframe=True)
