import torch
import sys
sys.path.append("./")
from models.pointnet2_utils import square_distance
import models.tsg_utils as tsg_utils

def distance_loss(pred_distance, sample_coord, gt_cent_coord, gt_cent_exist):
    # pred_distance: B, 16, encoder_sampled_points
    # sample_coord : B, 3, encoder_sampled_points
    # gt_centroid_coord : B, 3, 16
    # gt_cent_exist : B, 16
    B, _, _ = gt_cent_coord.shape
    gt_cent_exist = gt_cent_exist.view(gt_cent_exist.shape[0],gt_cent_exist.shape[1],1)

    sample_coord = sample_coord.permute(0,2,1)
    gt_cent_coord = gt_cent_coord.permute(0,2,1)

    # dists.shape -> B, 16, encoder_sampled_points
    dists = square_distance(gt_cent_coord, sample_coord)

    # dists.shape -> B, 16, enconder_sampled_points
    loss = torch.nn.functional.smooth_l1_loss(pred_distance, dists, reduction='none')
    loss = loss * gt_cent_exist
    loss = torch.div(torch.sum(loss), B)
    return loss

def centroid_dist_loss(pred_distance, pred_offset, sampled_coord, gt_cent_coord, gt_cent_exist):
    # pred_distance: B, 16, encoder_sampled_points
    # pred_offset : B, 48, encoder_sampled_points
    # sample_coord : B, 3, encoder_sampled_points
    # gt_centroid_coord : B, 3, 16
    # gt_cent_exist : B,16

    B, _, S_N = sampled_coord.shape

    gt_cent_coord = gt_cent_coord.permute(0,2,1)
    
    # 일단, 움직인 점을 찾자
    sampled_coord = sampled_coord.permute(0,2,1)

    #B, 1, encoder_sampeld_points, 3
    sampled_coord = sampled_coord.view(B, 1, S_N, 3)
    
    #B,16,3,ecnoder_sampeld_points
    pred_offset = pred_offset.view(B, 16, 3, S_N)
    #B,16,ecnoder_sampeld_points,3
    pred_offset = pred_offset.permute(0,1,3,2)

    #B, 16, encdoer_sampeld_points, 3
    moved_coord = sampled_coord + pred_offset

    losses = 0

    #B, 16, 1
    gt_cent_exist = gt_cent_exist.view(gt_cent_exist.shape[0],gt_cent_exist.shape[1],1)

    #B, 16 * encdoer_sampeld_points, 3 -> TODO
    for class_idx in range(16):
        #B, encoder_sampeled_points, 3
        moved_coord_in_class = moved_coord[:, class_idx, :, :]

        #B, 1, encdoer_sampeld_points
        dists = square_distance(gt_cent_coord[:, class_idx:class_idx+1, :], moved_coord_in_class)
        dists = dists.view(B, -1)

                        #B, 1 
        dists = dists * gt_cent_exist[:,class_idx,:]
        
        #B, encoder_sampled_points
        cls_pred_distance = pred_distance[:, class_idx, :]
        dists_mask = cls_pred_distance.le(0.03)
        fin_dists = torch.masked_select(dists, dists_mask)
        
        losses += torch.sum(fin_dists)
    losses = torch.div(losses, dists.shape[0])
    return losses

def exists_loss(pred_exist, gt_cent_exist):
    # pred_exist = B, 16
    # gt_cent_exist = B, 16
    gt_cent_exist = gt_cent_exist.type(torch.float)
    bce = torch.nn.BCEWithLogitsLoss()(pred_exist, gt_cent_exist)
    return bce

def centroid_loss(l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, exist_pred, gt_cent_coords, gt_cent_exists):
#l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, exist_pred
    dist_loss = distance_loss(dist_result, l3_xyz, gt_cent_coords, gt_cent_exists) 
    print("dist_loss",dist_loss)
    cent_dist_loss  = centroid_dist_loss(dist_result, offset_result, l3_xyz, gt_cent_coords, gt_cent_exists)
    print("cent_dist_loss",cent_dist_loss)
    ex_loss = exists_loss(exist_pred, gt_cent_exists) 
    print("ex_loss",ex_loss)
    return dist_loss + cent_dist_loss + ex_loss