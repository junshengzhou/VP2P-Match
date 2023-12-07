import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())
from models.pointTransformer.utils import Conv1d, PointNet_FP_Module, PointNet_SA_Module, Transformer # 


class PointsEncoder_label(nn.Module):
    def __init__(self, out_dim=512):
        super(PointsEncoder_label, self).__init__()

        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3, [64, 64, 128], group_all=False)
        self.transformer_start_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.transformer_start_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, out_dim], group_all=True)

    def forward(self, point_cloud, label):
        point_cloud = point_cloud.transpose(2,1).contiguous()

        label = label.unsqueeze(1)

        label = label.repeat(1,512,1)
        label = label.transpose(1,2).contiguous()

        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud
        b, _, n = l0_points.shape
        l1_xyz, l1_points_down = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
         
        l1_points_down = torch.cat((l1_points_down, label), dim=1) # + self.conv_label_onehot(label)  # label supervise

        l1_points_down = self.transformer_start_1(l1_points_down, l1_xyz)
        l2_xyz, l2_points_down = self.sa_module_2(l1_xyz, l1_points_down)  # (B, 3, 128), (B, 256, 128)
        l2_points_down = self.transformer_start_2(l2_points_down, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points_down)  # (B, 3, 1), (B, 1024, 1)


        return l3_points

class PointsEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super(PointsEncoder, self).__init__()

        self.sa_module_1 = PointNet_SA_Module(512, 32, 0.2, 3, [64, 64, 128], group_all=False)
        self.transformer_start_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module(128, 32, 0.4, 128, [128, 128, 256], group_all=False)
        self.transformer_start_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, out_dim], group_all=True)

    def forward(self, point_cloud):
        point_cloud = point_cloud.transpose(2,1).contiguous()

        device = point_cloud.device
        l0_xyz = point_cloud
        l0_points = point_cloud
        b, _, n = l0_points.shape
        l1_xyz, l1_points_down = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points_down = self.transformer_start_1(l1_points_down, l1_xyz)
        l2_xyz, l2_points_down = self.sa_module_2(l1_xyz, l1_points_down)  # (B, 3, 128), (B, 256, 128)
        l2_points_down = self.transformer_start_2(l2_points_down, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points_down)  # (B, 3, 1), (B, 1024, 1)


        return l3_points

class PointsEncoder_pointwise(nn.Module):
    def __init__(self, is_pc_norm):
        super(PointsEncoder_pointwise, self).__init__()

        if is_pc_norm:
            self.sa_module_1 = PointNet_SA_Module(1024, 64, 0.2, 7, [64, 64, 128], group_all=False) # radius x 100 0.2->20
            self.transformer_start_1 = Transformer(128, dim=64)
            self.sa_module_2 = PointNet_SA_Module(256, 32, 0.4, 128, [128, 128, 256], group_all=False)
            self.transformer_start_2 = Transformer(256, dim=64)
            self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 512], group_all=True)
        else:
            self.sa_module_1 = PointNet_SA_Module(1024, 64, 20, 7, [64, 64, 128], group_all=False) # radius x 100 0.2->20
            self.transformer_start_1 = Transformer(128, dim=64)
            self.sa_module_2 = PointNet_SA_Module(256, 32, 40, 128, [128, 128, 256], group_all=False)
            self.transformer_start_2 = Transformer(256, dim=64)
            self.sa_module_3 = PointNet_SA_Module(None, None, None, 256, [256, 512, 512], group_all=True)

        self.fp_module_3 = PointNet_FP_Module(512, [256, 256], use_points1=True, in_channel_points1=256)
        self.transformer_end_2 = Transformer(256, dim=64)
        self.fp_module_2 = PointNet_FP_Module(256, [256, 128], use_points1=True, in_channel_points1=128)
        self.transformer_end_1 = Transformer(128, dim=64)
        self.fp_module_1 = PointNet_FP_Module(128, [128, 128, 128], use_points1=True, in_channel_points1=6)

        self.conv_DownFeatureStep1 = Conv1d(128, 128, if_bn=True, activation_fn=torch.relu)
        self.conv_DownFeatureStep2 = Conv1d(256, 256, if_bn=True, activation_fn=torch.relu)

    def forward(self, point_cloud):

        device = point_cloud.device
        l0_xyz = point_cloud[:,0:3,:].contiguous()
        l0_points = point_cloud
        b, _, n = l0_points.shape
        l1_xyz, l1_points_down = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
         
        l1_points_down = self.transformer_start_1(l1_points_down, l1_xyz)
        l2_xyz, l2_points_down = self.sa_module_2(l1_xyz, l1_points_down)  # (B, 3, 128), (B, 256, 128)
        l2_points_down = self.transformer_start_2(l2_points_down, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points_down)  # (B, 3, 1), (B, 1024, 1)

        l2_points_up = self.fp_module_3(l2_xyz, l3_xyz, l2_points_down, l3_points)
        l2_points_down = self.conv_DownFeatureStep2(l2_points_down)
        l2_points_up = l2_points_up + l2_points_down
        l2_points_up = self.transformer_end_2(l2_points_up, l2_xyz)

        l1_points_up = self.fp_module_2(l1_xyz, l2_xyz, l1_points_down, l2_points_up)
        l1_points_down = self.conv_DownFeatureStep1(l1_points_down)
        l1_points_up = l1_points_up + l1_points_down
        l1_points_up = self.transformer_end_1(l1_points_up, l1_xyz)

        l0_points = self.fp_module_1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_xyz], 1), l1_points_up)

        return l3_points.squeeze(2), l0_points
