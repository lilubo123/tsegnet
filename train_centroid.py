import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import os
import numpy as np
from models import tsg_centroid_module, tsg_seg_module
from models.tsg_loss import centroid_loss, segmentation_loss, segmentation_mask_loss
from torch.optim.lr_scheduler import ExponentialLR
import models.tsg_utils as utils
from models.generator import CenterPointGenerator

cuda = torch.device('cuda')

centroid_model = tsg_centroid_module.get_model()
centroid_model.cuda()

optimizer = torch.optim.Adam(centroid_model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, 0.999)

point_loader = DataLoader(CenterPointGenerator(), batch_size=1)
val_point_loader = DataLoader(CenterPointGenerator("data/sampled_val"), batch_size=1)
best_loss = -100
for epoch in range(10000):
    centroid_model.train()

    total_loss = 0
    for batch_item in point_loader:
        points = batch_item[0].cuda()
        centroids = batch_item[1].cuda()
        y_center_pred = centroid_model(points)

        loss = centroid_loss(y_center_pred[4], y_center_pred[3], y_center_pred[5], centroids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss
    print("train_loss", total_loss)
    torch.save(centroid_model.state_dict(), "pretrained_cent_model_train.h5")
    #val
    centroid_model.eval()
    
    total_val_loss = 0
    for batch_item in val_point_loader:
        points = batch_item[0].cuda()
        centroids = batch_item[1].cuda()
        y_center_pred = centroid_model(points)
        with torch.no_grad():
            val_loss = centroid_loss(y_center_pred[4], y_center_pred[3], y_center_pred[5], centroids)
        total_val_loss += val_loss
    print("total_val_loss", total_val_loss)
    if total_val_loss < best_loss:
        best_loss = total_val_loss
        torch.save(centroid_model.state_dict(), "pretrained_cent_model.h5")
