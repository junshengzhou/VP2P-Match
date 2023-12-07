import torchsparse
import torchsparse.nn as spnn
from torch import nn
from torchsparse import PointTensor
import torch

from models.spvnas.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN']


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.num_classes = kwargs['num_classes']

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(7, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)
        self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
        # self.pool2 = lambda x: torch.max(x, dim=1, keepdim=False)[0]

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats, corrds, batch_size):
        # x: SparseTensor z: PointTensor
        z = PointTensor(feats, corrds.float())

        x0 = initial_voxelize(z, self.pres, self.vres)
        # print(feats.shape, x0.F.shape)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F
        # print(z0.F.shape)

        x1 = point_to_voxel(x0, z0)
        # print(x1.F.shape)

        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # print(x1.F.shape, x2.F.shape, x3.F.shape, x4.F.shape)
        z1 = voxel_to_point(x4, z0)
        # print(z1.F.shape)
        
        # print(z1.F.shape, torch.unique(z1.F, dim=0).shape)
        # print(feats[:,:3].shape, torch.unique(feats[:,:3], dim=0).shape)
        # print(torch.unique(z1.F, dim=0, sorted=True, return_counts=True))
        # for tmp in z1.F:
        #     # print(tmp.shape, torch.unique(z1.F, dim=0, sorted=True)[0].shape)
        #     if tmp.equal(torch.unique(z1.F, dim=0, sorted=True)[1]):
        #         print(tmp)
        # # sort_F = torch.sort(z1.F, dim=0)
        # # print(sort_F[0][:100])
        # exit()

        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        # print(y1.F.shape)
        y1 = self.up1[0](y1)
        # print(y1.F.shape)
        y1 = torchsparse.cat([y1, x3])
        # print(y1.F.shape)
        y1 = self.up1[1](y1)
        # print(y1.F.shape)
        # exit()
        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        # print(z3.F.shape)
        # exit()


        out = self.classifier(z3.F)
        pointwised_feat = out.reshape(batch_size, -1, self.num_classes).transpose(2,1)
        global_feat = self.pool(pointwised_feat).squeeze(-1)
        # print(global_feat.shape, pointwised_feat.shape)

        return global_feat, pointwised_feat

    # def forward(self, feats, corrds, batch_size):
    #     # x: SparseTensor z: PointTensor
    #     z = PointTensor(feats, corrds.float())
    #     z = z.F#.unsqueeze(0).transpose(2,1)
    #     # print(z.shape)
    #     z = self.point_transforms[0](z)
    #     z = self.point_transforms[1](z)
    #     z = self.point_transforms[2](z)


    #     out = self.classifier(z)
    #     print(out.shape)
    #     pointwised_feat = out.reshape(batch_size, -1, self.num_classes).transpose(2,1)
    #     print(pointwised_feat.shape)
    #     global_feat = self.pool(pointwised_feat).squeeze(1)
    #     print(global_feat.shape)
    #     # print(global_feat.shape, pointwised_feat.shape)

    #     return global_feat, pointwised_feat

if __name__ == '__main__':
    model = SPVCNN(num_classes=10, cr=0.5, pres=0.05, vres=0.05)
