import torch
import sys
import importlib
import models.tsg_loss as tsg_loss
import models
from models import tsg_centroid_module, tsg_seg_module
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import numpy as np
from models.tsg_centroid_module import get_model
from models.tsg_loss import centroid_loss
from torch.optim.lr_scheduler import ExponentialLR
from models.pointnet2_utils import square_distance
from models.generator import CenterPointGenerator
import models.tsg_utils as utils
from models import gen_utils
    
centroid_model = models.tsg_centroid_module.get_model()
#centroid_model.load_state_dict(torch.load("pretrained_centroid_model.h5"))
centroid_model.load_state_dict(torch.load("model_centroid_recent_train"))
centroid_model.cuda()
centroid_model.eval()

point_loader = DataLoader(CenterPointGenerator(), batch_size=1)



seg_model = tsg_seg_module.get_model()
#seg_model.load_state_dict(torch.load("model_seg"))
seg_model.load_state_dict(torch.load("model_segmentation_recent_train"))
seg_model.cuda()
seg_model.eval()


#============centroid model===========#
for item in point_loader:
    points = item[0].cuda()
    centroids = item[1].cuda()
    seg_label = item[2].cuda()
    with torch.no_grad():
        centroid_model_output = centroid_model(points)

cen_cpu = centroids.cpu().detach().numpy()[0,:].T
global_labeld_map_gt = np.concatenate([points[0][:3].cpu().detach().numpy().T, seg_label[0][:1].cpu().detach().numpy().T], axis=1)
o3d.visualization.draw_geometries([gen_utils.np_to_by_label(global_labeld_map_gt, axis=3), gen_utils.np_to_pcd(cen_cpu)], mesh_show_back_face=False,mesh_show_wireframe=True)

#==============crop processing==================#
sampled_db_scan = utils.dbscan_pc(centroid_model_output[3], centroid_model_output[4], centroid_model_output[5])
#sampled_db_scan = [centroids[0].cpu().detach().numpy().T]
nearest_n = utils.get_nearest_neighbor_idx(centroid_model_output[2], sampled_db_scan, 1024)
cropped_coords, sampled_centroids = utils.get_nearest_neighbor_points_with_centroids(centroid_model_output[2], nearest_n, sampled_db_scan)
cropped_features, _ = utils.get_nearest_neighbor_points_with_centroids(centroid_model_output[0], nearest_n, sampled_db_scan)
cropped_gt_labels, _ = utils.get_nearest_neighbor_points_with_centroids(seg_label, nearest_n, sampled_db_scan)
print(cropped_coords.shape)
print(cropped_features.shape)
seg_input = utils.concat_seg_input(cropped_features, cropped_coords, sampled_centroids)

#===============seg model===============#
with torch.no_grad():
    seg_model_output = seg_model(seg_input)



#==============cropped 마다 보이기===================#
for i in range(seg_model_output[0].shape[0]):
    test_cropped = cropped_coords.cpu().detach().numpy()[i].T[:,:3]
    test_label = seg_model_output[2].cpu().detach().numpy()[i].T.reshape(-1,1)
    test_weight = seg_model_output[1].cpu().detach().numpy()[i].T.reshape(-1,1)
    test_label = gen_utils.sigmoid(test_label)# * gen_utils.sigmoid(test_weight)
    test_label[test_label>0.5] = 1
    test_label[test_label<0.5] = 0
    test_label = test_label.astype(int)
    test_cropped = np.concatenate([test_cropped, test_label],axis=1)
    o3d.visualization.draw_geometries([gen_utils.np_to_by_label(test_cropped, axis=3), gen_utils.np_to_pcd(cen_cpu)], mesh_show_back_face=False,mesh_show_wireframe=True)

#=================전체 포인트 클라우드에서 보이기===================#
global_labeled_map = gen_utils.cropped_to_global_label_marker(points[0][:3], seg_model_output[2], seg_model_output[1], seg_model_output[3], nearest_n[0])
o3d.visualization.draw_geometries([gen_utils.np_to_by_label(global_labeled_map, axis=3), gen_utils.np_to_pcd(cen_cpu)], mesh_show_back_face=False,mesh_show_wireframe=True)
