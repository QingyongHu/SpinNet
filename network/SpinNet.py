import sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import network.ThreeDCCN as pn
import script.common as cm
from script.common import switch


class Descriptor_Net(nn.Module):
    def __init__(self, des_r, rad_n, azi_n, ele_n, voxel_r, voxel_sample, dataset):
        super(Descriptor_Net, self).__init__()
        self.des_r = des_r
        self.rad_n = rad_n
        self.azi_n = azi_n
        self.ele_n = ele_n
        self.voxel_r = voxel_r
        self.voxel_sample = voxel_sample
        self.dataset = dataset

        self.bn_xyz_raising = nn.BatchNorm2d(16)
        self.bn_mapping = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()
        self.xyz_raising = nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        self.conv_net = pn.Cylindrical_Net(inchan=16, dim=32)

    def forward(self, input):
        center = input[:, -1, :].unsqueeze(1)
        delta_x = input[:, :, 0:3] - center[:, :, 0:3]  # (B, npoint, 3), normalized coordinates
        for case in switch(self.dataset):
            if case('3DMatch'):
                z_axis = cm.cal_Z_axis(delta_x, ref_point=input[:, -1, :3])
                z_axis = cm.l2_norm(z_axis, axis=1)
                R = cm.RodsRotatFormula(z_axis, torch.FloatTensor([0, 0, 1]).unsqueeze(0).repeat(z_axis.shape[0], 1))
                delta_x = torch.matmul(delta_x, R)
                break
            if case('KITTI'):
                break

        # partition the local surface along elevator, azimuth, radial dimensions
        S2_xyz = torch.FloatTensor(cm.get_voxel_coordinate(radius=self.des_r,
                                                           rad_n=self.rad_n,
                                                           azi_n=self.azi_n,
                                                           ele_n=self.ele_n))

        pts_xyz = S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).cuda()
        # query points in sphere
        new_points = cm.sphere_query(delta_x, pts_xyz, radius=self.voxel_r,
                                     nsample=self.voxel_sample)
        # transform rotation-variant coords into rotation-invariant coords
        new_points = new_points - pts_xyz.unsqueeze(2).repeat([1, 1, self.voxel_sample, 1])
        new_points = cm.var_to_invar(new_points, self.rad_n, self.azi_n, self.ele_n)

        new_points = new_points.permute(0, 3, 1, 2)  # (B, C_in, npoint, nsample), input features
        C_in = new_points.size()[1]
        nsample = new_points.size()[3]
        x = self.activation(self.bn_xyz_raising(self.xyz_raising(new_points)))
        x = F.max_pool2d(x, kernel_size=(1, nsample)).squeeze(3)  # (B, C_in, npoint)
        del new_points
        del pts_xyz
        x = x.view(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)

        x = self.conv_net(x)
        x = F.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]))

        return x

    def get_parameter(self):
        return list(self.parameters())
