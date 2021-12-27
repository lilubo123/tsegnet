import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from models.tsg_loss import centroid_loss

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        
        input_feauture_num = 6
        # target point 개수, ball query radius, maximun sample in ball 개수, input feature 개수(position + 각각의 feature vector), MLP 개수, group_all False 
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.0025, 0.005], [16, 32], input_feauture_num, [[32, 32], [32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.005, 0.01], [16, 32], 32+32, [[64, 128], [64, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.01, 0.02], [16, 32], 128+128, [[196, 256], [196, 256]])
        

        self.fp3 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1 = PointNetFeaturePropagation(128+input_feauture_num, [64, 32])

        self.offset_conv_1 = nn.Conv1d(512,256, 1)
        self.offset_bn_1 = nn.BatchNorm1d(256)
        self.dist_conv_1 = nn.Conv1d(512,256, 1)
        self.dist_bn_1 = nn.BatchNorm1d(256)
        
        self.offset_conv_2 = nn.Conv1d(256,3, 1)
        self.dist_conv_2 = nn.Conv1d(256,1, 1)

        nn.init.zeros_(self.offset_conv_2.weight)
        nn.init.zeros_(self.dist_conv_2.weight)

        #prediction part
        self.conv1 = nn.Conv1d(32, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
    #input으로, batch, channel(xyz + 기타등등), sample in batch 이렇게 와야 한다.
    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))

        offset_result = F.relu(self.offset_bn_1(self.offset_conv_1(l3_points)))
        offset_result = self.offset_conv_2(offset_result)

        dist_result = F.relu(self.dist_bn_1(self.dist_conv_1(l3_points)))
        dist_result = self.dist_conv_2(dist_result)
        
        return x, l3_points, l0_xyz, l3_xyz, offset_result, dist_result


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import torch
    model = get_model()
    xyz = torch.rand(6, 6, 2048)
    #output is B, C, N order
    for item in model(xyz):
        print(item.shape)
    input()