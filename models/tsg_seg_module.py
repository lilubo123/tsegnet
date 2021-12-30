import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation,PointNetSetAbstraction
import torch
class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        
        # target point 개수, ball query radius, maximun sample in ball 개수, input feature 개수(position + 각각의 feature vector), MLP 개수, group_all False 
        #first module
        input_feature_num = 36
        self.sa1_1 = PointNetSetAbstractionMsg(1024, [0.0025, 0.005], [16, 32], input_feature_num, [[32, 32], [32, 32]])
        self.sa2_1 = PointNetSetAbstractionMsg(512, [0.005, 0.01], [16, 32], 32+32, [[64, 128], [64, 128]])
        self.sa3_1 = PointNetSetAbstractionMsg(256, [0.01, 0.02], [16, 32], 128+128, [[196, 256], [196, 256]])
        

        self.fp3_1 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2_1 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1_1 = PointNetFeaturePropagation(128+input_feature_num, [64, 32])

        self.pd_mask_1 = nn.Conv1d(32,2,1)
        self.pd_mask_1_softmax = torch.nn.Softmax(dim=1)
        self.wt_mask_1 = nn.Conv1d(32,1,1)
        #second module
        second_feature_num = 38
        self.sa1_2 = PointNetSetAbstractionMsg(1024, [0.0025, 0.005], [16, 32], second_feature_num, [[32, 32], [32, 32]])
        self.sa2_2 = PointNetSetAbstractionMsg(512, [0.005, 0.01], [16, 32], 32+32, [[64, 128], [64, 128]])
        self.sa3_2 = PointNetSetAbstractionMsg(256, [0.01, 0.02], [16, 32], 128+128, [[196, 256], [196, 256]])
        self.flatten_sa = PointNetSetAbstraction(None, None, None, 512+3, [256, 512], True)

        self.fp3_2 = PointNetFeaturePropagation(768, [256, 256])
        self.fp2_2 = PointNetFeaturePropagation(320, [128, 128])
        self.fp1_2 = PointNetFeaturePropagation(128+second_feature_num, [64, 32])

        self.pd_mask_2 = nn.Conv1d(32,1,1)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.LayerNorm(256)#nn.BatchNorm1d(256)
        #self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 8)
        #self.drop2 = nn.Dropout(0.5)

    #input으로, batch, channel(xyz + 기타등등), sample in batch 이렇게 와야 한다.
    def forward(self, xyz):
        B, _, _ = xyz.shape

        l0_points_0 = xyz
        l0_xyz_0 = xyz[:,:3,:]

        l1_xyz_0, l1_points_0 = self.sa1_1(l0_xyz_0, l0_points_0)
        l2_xyz_0, l2_points_0 = self.sa2_1(l1_xyz_0, l1_points_0)
        l3_xyz_0, l3_points_0 = self.sa3_1(l2_xyz_0, l2_points_0)

        l2_points_0 = self.fp3_1(l2_xyz_0, l3_xyz_0, l2_points_0, l3_points_0)
        l1_points_0 = self.fp2_1(l1_xyz_0, l2_xyz_0, l1_points_0, l2_points_0)
        l0_points_0= self.fp1_1(l0_xyz_0, l1_xyz_0, l0_points_0, l1_points_0)

        pd_1 = self.pd_mask_1_softmax(self.pd_mask_1(l0_points_0))
        weight_1 = self.wt_mask_1(l0_points_0)

        l0_points_1 = torch.cat([xyz, pd_1],dim=1)
        l0_xyz_1 = xyz[:,:3,:]
        
        l1_xyz_1, l1_points_1 = self.sa1_2(l0_xyz_1, l0_points_1)
        l2_xyz_1, l2_points_1 = self.sa2_2(l1_xyz_1, l1_points_1)
        l3_xyz_1, l3_points_1 = self.sa3_2(l2_xyz_1, l2_points_1)

        l2_points_1 = self.fp3_2(l2_xyz_1, l3_xyz_1, l2_points_1, l3_points_1)
        l1_points_1 = self.fp2_2(l1_xyz_1, l2_xyz_1, l1_points_1, l2_points_1)
        l0_points_1= self.fp1_2(l0_xyz_1, l1_xyz_1, l0_points_1, l1_points_1)
        
        _, id_pred = self.flatten_sa(l3_xyz_1, l3_points_1)
        id_pred = id_pred.view(B, 512)
        id_pred = F.relu(self.bn1(self.fc1(id_pred)))
        id_pred = self.fc2(id_pred)

        pd_2 = self.pd_mask_2(l0_points_1)
        
        return pd_1, weight_1, pd_2, id_pred


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import torch
    model = get_model()
    model.cuda()
    
    xyz = torch.rand(4, 36, 4096).cuda()
    #output is B, C, N order
    for item in model(xyz):
        print(item.shape)
    input()