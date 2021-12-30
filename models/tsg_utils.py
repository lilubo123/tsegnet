from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neighbors import KDTree
import torch 
from models.pointnet2_utils import square_distance

def dbscan_pc(sample_xyz, pred_offset, pred_dist):
    #pred_centroids = B, C, N
    #pred_dist = B,N
    #output = B, N, C
    pred_centroids = sample_xyz + pred_offset
    pred_centroids = pred_centroids.permute(0,2,1).cpu().detach().numpy()
    pred_dist = pred_dist.cpu().detach().numpy()


    sampled_centroids = []
    for batch_idx in range(pred_centroids.shape[0]):
        batch_pred_centroids = pred_centroids[batch_idx,:,:]
        batch_pred_dist = pred_dist[batch_idx, :]
        batch_pred_centroids = batch_pred_centroids[(batch_pred_dist<0.015).reshape(-1)]
        clustering = DBSCAN(eps=0.02, min_samples=2).fit(batch_pred_centroids,3)
        unique_labels = np.unique(clustering.labels_)
        clustering_ls = []
        for label in unique_labels:
            if(label != -1):
                clustering_ls.append(np.mean(batch_pred_centroids[clustering.labels_==label],axis=0))
        sampled_centroids.append(clustering_ls)
    return sampled_centroids

def get_nearest_neighbor_idx(org_xyz, sampled_clusters, crop_num=4096):
    #org_xyz -> B, 3, N
    #sampled_clusters -> B, cluster_num, 3
    #return - B, cluster_num, 4096 
    org_xyz = org_xyz.permute(0,2,1).cpu().detach().numpy()
    cropped_all = []
    for batch_idx in range(org_xyz.shape[0]):
        cropped_points = []

        tree = KDTree(org_xyz[batch_idx,:,:], leaf_size=2)
        for cluster_centroid in sampled_clusters[batch_idx]:
            indexs = tree.query([cluster_centroid], k=crop_num, return_distance = False)
            cropped_points.append(indexs[0])
        cropped_all.append(cropped_points)
    return cropped_all

def get_nearest_neighbor_points_with_centroids(feature_xyz, cropped_all_indexes, sampled_db_scan):
    #org_xyz -> 1(방법이 없다.), features, N
    #cropped_all_indexes -> 1, centroid_num, 4096
    #sampled_clusters -> 1, cluster_num, 3
    #return -> centroid_num(어쩔수없다), features, 4096 // centroid_num, 3, 1
    #돌아가면서 for문으로 빼는 방식으로 만들 것,
    for b_idx in range(len(cropped_all_indexes)):
        cropped_points = []
        for cluster_idx in range(len(cropped_all_indexes[0])):
            cropped_point = torch.index_select(feature_xyz[0,:,:], 1, torch.tensor(cropped_all_indexes[b_idx][cluster_idx]).cuda())
            cropped_points.append(cropped_point)
            
        cropped_points = torch.stack(cropped_points, dim=0)

        sampled_clusters_cuda = torch.from_numpy(np.array(sampled_db_scan[0])).cuda()
        sampled_clusters_cuda = sampled_clusters_cuda.view((-1,3,1))
        return cropped_points, sampled_clusters_cuda
    
def get_nearest_neighbor_points_hold_batch(feature_xyz, cropped_all_indexes, sampled_db_scan, rand_cluster_indexes):
    #org_xyz -> B, features, N
    #cropped_all_indexes -> B, centroid_num, 4096
    #sampled_clusters -> B, cluster_num, 3
    #cluster_idx -> B : int
    #random으로 하나의 teeth만 고른다.
    #return -> B, features, 4096 // B, 3, 1
    #돌아가면서 for문으로 빼는 방식으로 만들 것,
    cropped_points = []
    sampled_clusters_seleceted = []
    for b_idx in range(len(cropped_all_indexes)):
        cropped_point = torch.index_select(feature_xyz[0,:,:], 1, torch.tensor(cropped_all_indexes[b_idx][rand_cluster_indexes[b_idx]]).cuda())
        cropped_points.append(cropped_point)
        sampled_clusters_seleceted.append(sampled_db_scan[b_idx][rand_cluster_indexes[b_idx]])
    cropped_points = torch.stack(cropped_points, dim=0)
    sampled_clusters_cuda = torch.from_numpy(np.array(sampled_clusters_seleceted)).cuda()
    sampled_clusters_cuda = sampled_clusters_cuda.view((-1,3,1))
    return cropped_points, sampled_clusters_cuda
        
def concat_seg_input(cropped_features, cropped_coord, sampled_centroid_points_cuda):
    # cropped_features : B(crooped 개수), 16, 4096
    # cropped_coord : B(crooped 개수), 3, 4096
    # sampled_centroid_points_cuda : B, 3, 1
    cropped_coord = cropped_coord.permute(0,2,1)
    sampled_centroid_points_cuda = sampled_centroid_points_cuda.permute(0,2,1)
    
    ddf = square_distance(cropped_coord, sampled_centroid_points_cuda)
    ddf *= (-4)
    ddf = torch.exp(ddf)

    cropped_coord = cropped_coord.permute(0,2,1)
    ddf = ddf.permute(0,2,1)
    concat_result = torch.cat([cropped_coord, cropped_features, ddf], dim=1)
    return concat_result

def get_gt_labels_maximum(gt_label):
    # gt_label: batch_size, , cropped
    # gt_bin_label: batch_size, 1, cropped
    gt_max_labels = []
    proposal_gt_label = gt_label.view(gt_label.shape[0], -1)
    for batch_idx in range(proposal_gt_label.shape[0]):
        label_indexes, counts = proposal_gt_label[batch_idx, :].unique(sorted=True, return_counts = True)
        max_index = torch.argmax(counts[1:])
        max_label = torch.index_select(label_indexes[1:], 0, max_index)
        gt_max_labels += [max_label]
    gt_max_labels = torch.stack(gt_max_labels, dim=0)
    gt_max_labels = gt_max_labels.view(-1,1)
    
    gt_label = gt_label.view(gt_label.shape[0],-1)
    gt_bin_label = torch.zeros_like(gt_label)
    gt_bin_label[gt_label == gt_max_labels] = 1
    gt_bin_label = gt_bin_label.type(torch.long)
    
    return gt_bin_label

def get_cluster_gt_id_by_nearest_gt_centroid(gt_centorids_coords, gt_centroids_ids, pred_centroid_coords):
    # gt_centorids_coords batch_size, 3, num of gt centroid, 모든 클러스터에 대해 , 배치마다!
    # gt_centroids_ids batch_size, 1, num of gt centroid, 모든 클러스터에 대해 , 배치마다!
    # pred_centroid_coords batch_size, 3, 1
    # pred_cluster_gt_id batch_size, 1, 1 

    gt_centorids_coords = gt_centorids_coords.permute(0,2,1)
    pred_centroid_coords = pred_centroid_coords.permute(0,2,1)

    nearby_centroids_dists = square_distance(pred_centroid_coords, gt_centorids_coords)
    _, nearby_centroids_indexes = nearby_centroids_dists.sort(dim=-1)
    nearby_centroids_indexes = nearby_centroids_indexes[:,:,0]
    pred_cluster_gt_ids = []
    for batch_idx in range(nearby_centroids_indexes.shape[0]):
        pred_cluster_gt_ids.append(gt_centroids_ids[batch_idx,0,nearby_centroids_indexes[batch_idx]])
    pred_cluster_gt_ids = torch.stack(pred_cluster_gt_ids, dim=0)
    return pred_cluster_gt_ids.view(-1,1,1)

def get_cluster_points_bin_label(gt_point_labels_inside_cluster, pred_cluster_gt_ids):
    # gt_point_labels_inside_cluster batch_size, 1, sampling num(4096)
    # cluster_points_labels batch_size, 1, num of points inside cluster
    # pred_cluster_gt_id batch_size, 1, 1 
    gt_labels = gt_point_labels_inside_cluster.view(gt_point_labels_inside_cluster.shape[0], gt_point_labels_inside_cluster.shape[2])
    centroid_labels = pred_cluster_gt_ids.view(pred_cluster_gt_ids.shape[0], pred_cluster_gt_ids.shape[2])
    gt_bin_label = torch.zeros_like(gt_labels)
    gt_bin_label[gt_labels == centroid_labels] = 1
    gt_bin_label = gt_bin_label.type(torch.long)
    gt_bin_label = gt_bin_label.view(gt_labels.shape[0], 1, gt_labels.shape[1])

    return gt_bin_label

def get_gt_labels_nearest_points(cropped_coord, centroids, gt_labels):
    # 기각,,,
    # cropped_coords: batch_size, 3, num of points inside cluster 
    # centroid : batch_size, 3, 1
    # gt_labels: batch_size, 1(몇번인지), num of points inside cluster
    
    # gt_bin_label: batch_size, 1, cropped
    print(cropped_coord.shape)
    print(centroids.shape)
    centroids = centroids.permute(0,2,1)
    cropped_coord = cropped_coord.permute(0,2,1)

    nearby_points_dists = square_distance(centroids, cropped_coord)
    _, sorted_nearby_points_indexes = nearby_points_dists.sort(dim=-1)
    sorted_nearby_points_indexes = sorted_nearby_points_indexes.view(sorted_nearby_points_indexes.shape[0],sorted_nearby_points_indexes.shape[2])
    gt_near_labels = []

    for batch_idx in range(sorted_nearby_points_indexes.shape[0]):
        flat_gt_labels = gt_labels[batch_idx,:,:].view(-1)
        label_indexes, counts = flat_gt_labels[sorted_nearby_points_indexes[batch_idx, :100]].unique(sorted=True, return_counts = True)
        
        max_index = torch.argmax(counts[1:])
        max_label = torch.index_select(label_indexes[1:], 0, max_index)
        gt_near_labels += [max_label]
    gt_near_labels = torch.stack(gt_near_labels, dim=0)
    gt_near_labels = gt_near_labels.view(-1,1)

    gt_label = gt_labels.view(gt_labels.shape[0],-1)
    gt_bin_label = torch.zeros_like(gt_label)
    gt_bin_label[gt_label == gt_near_labels] = 1
    gt_bin_label = gt_bin_label.type(torch.long)
    return gt_bin_label