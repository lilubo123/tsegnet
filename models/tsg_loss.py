import torch
import sys
sys.path.append("./")
from models.pointnet2_utils import square_distance

def distance_loss(pred_distance, sample_xyz, centroid):
    pred_distance = pred_distance.view(-1, sample_xyz.shape[2])
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    dists = square_distance(sample_xyz, centroid)
    sorted_dists, _ = dists.sort(dim=-1)
    min_dists = sorted_dists[:, :, 0]
    loss = torch.nn.functional.smooth_l1_loss(pred_distance, min_dists)
    return loss

def centroid_dist_loss(pred_offset, sample_xyz, distance, centroid):
    distance = distance.view(-1, sample_xyz.shape[2])
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(pred_offset, sample_xyz)

    #source를 pred centroid로
    pred_ct_dists = square_distance(pred_centroid, centroid)
    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, 0]
    pred_ct_mask = distance.le(0.02)
    fin_pred_ct_dists = torch.masked_select(min_pred_ct_dists, pred_ct_mask)
    loss = torch.sum(fin_pred_ct_dists)

    #source를 centroid로
    ct_dists = square_distance(centroid, pred_centroid)
    sorted_ct_dists, _ = ct_dists.sort(dim=-1)
    min_ct_dists = sorted_ct_dists[:, :, 0]
    ct_mask = min_ct_dists.le(0.02)
    fin_ct_dists = torch.masked_select(min_ct_dists, ct_mask)
    loss += torch.sum(fin_ct_dists)
    return loss

def chamfer_distance_loss(pred_offset, sample_xyz, centroid):
    pred_offset = pred_offset.permute(0,2,1)
    sample_xyz = sample_xyz.permute(0,2,1)
    centroid = centroid.permute(0,2,1)

    pred_centroid = torch.add(pred_offset, sample_xyz)

    #source를 pred centroid로
    pred_ct_dists = square_distance(pred_centroid, centroid)
    sorted_pred_ct_dists, _ = pred_ct_dists.sort(dim=-1)
    min_pred_ct_dists = sorted_pred_ct_dists[:, :, :2]

    pred_ct_mask = min_pred_ct_dists[:,:,1].le(0.02)
    
    ratio = torch.div(min_pred_ct_dists[:,:,0], min_pred_ct_dists[:,:,1])
    ratio = torch.masked_select(ratio, pred_ct_mask)
    
    loss = torch.sum(ratio)
    return loss

def centroid_loss(pred_offset, sample_xyz, distance, centroid):
    loss = distance_loss(distance, sample_xyz, centroid)
    loss += centroid_dist_loss(pred_offset, sample_xyz, distance, centroid)
    loss += (chamfer_distance_loss(pred_offset, sample_xyz, centroid) * 0.1 )
    return loss

def first_seg_loss(pred_)