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
            indexs = tree.query([cluster_centroid], k=4096, return_distance = False)
            cropped_points.append(indexs[0])
        cropped_all.append(cropped_points)
    return cropped_all

def get_nearest_neighbor_points_with_centroids(feature_xyz, cropped_all_indexes, sampled_clusters):
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

        sampled_clusters_cuda = torch.from_numpy(np.array(sampled_clusters[0])).cuda()
        sampled_clusters_cuda = sampled_clusters_cuda.view((-1,3,1))
        return cropped_points, sampled_clusters_cuda

def concat_seg_input(cropped_features, cropped_coord, sampled_centroid_points_cuda):
    # cropped_features : B(crooped 개수), 16, 4096
    # cropped_coord : B(crooped 개수), 3, 4096
    # sampled_centroid_points_cuda : centroid_num, 3, 1
    cropped_coord = cropped_coord.permute(0,2,1)
    sampled_centroid_points_cuda = sampled_centroid_points_cuda.permute(0,2,1)
    
    ddf = square_distance(cropped_coord, sampled_centroid_points_cuda)
    ddf *= (-4)
    ddf = torch.exp(ddf)

    cropped_coord = cropped_coord.permute(0,2,1)
    ddf = ddf.permute(0,2,1)
    concat_result = torch.cat([cropped_coord, cropped_features, ddf], dim=1)
    return concat_result