import sys
import os
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import numpy as np
from models import tsg_centroid_classify_full_module
from models.tsg_centroid_classify_full_loss import centroid_loss
from torch.optim.lr_scheduler import ExponentialLR
import models.tsg_utils as utils
from models.generator_classify_full import CenterPointGenerator

cuda = torch.device('cuda')

centroid_model = tsg_centroid_classify_full_module.get_model()
centroid_model.cuda()

optimizer = torch.optim.Adam(centroid_model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, 0.999)

point_loader = DataLoader(CenterPointGenerator(), batch_size=2)
val_point_loader = DataLoader(CenterPointGenerator("data/sampled_cls_full_val"), batch_size=2)
best_loss = 100000
for epoch in range(10000):
    centroid_model.train()

    total_loss = 0
    for batch_item in point_loader:
        points = batch_item[0].cuda()
        gt_centroid_coords = batch_item[1].cuda()
        gt_centroid_exists = batch_item[3].cuda()
        y_center_pred = centroid_model(points)
#        return l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, exist_pred
        loss = centroid_loss(*y_center_pred, gt_centroid_coords, gt_centroid_exists)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss
    print("train_loss", total_loss)
    torch.save(centroid_model.state_dict(), "ckpt_cls/cls_full16/0103_train.h5")


    #val
    total_val_loss = 0

    for batch_item in point_loader:
        points = batch_item[0].cuda()
        gt_centroid_coords = batch_item[1].cuda()
        gt_centroid_exists = batch_item[3].cuda()
        with torch.no_grad():
            y_center_pred = centroid_model(points)
#        return l0_points, l3_points, l0_xyz, l3_xyz, offset_result, dist_result, exist_pred
        total_val_loss += centroid_loss(*y_center_pred, gt_centroid_coords, gt_centroid_exists)

    print("total_val_loss", total_val_loss)
    if total_val_loss < best_loss:
        best_loss = total_val_loss
        torch.save(centroid_model.state_dict(), "ckpt_cls/cls_full16/0103_val.h5")