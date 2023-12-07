import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, in_channels, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []

        x = self.conv1(x)  # /2
        x = self.bn1(x)
        x = self.relu(x)
        out.append(x)

        x = self.maxpool(x)

        x = self.layer1(x)  # /4
        out.append(x)
        x = self.layer2(x)  # /8
        out.append(x)
        x = self.layer3(x)  # /16
        out.append(x)
        x = self.layer4(x)  # /32
        out.append(x)
        x = self.avgpool(x)  # Cx1x1
        out.append(x)

        return out


def _resnet(in_channels, arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(in_channels, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels, 'resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(in_channels, 'resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(in_channels, 'resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(in_channels, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(in_channels, 'wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(in_channels=3, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(in_channels, 'wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)





class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        
        self.backbone = resnet34(in_channels=3, pretrained=False, progress=True)

        # image mesh grid
        '''input_mesh_np = np.meshgrid(np.linspace(start=0, stop=self.opt.img_W - 1, num=self.opt.img_W),
                                    np.linspace(start=0, stop=self.opt.img_H - 1, num=self.opt.img_H))
        input_mesh = torch.from_numpy(np.stack(input_mesh_np, axis=0).astype(np.float32)).to(self.opt.device)  # 2xHxW
        self.input_mesh = input_mesh.unsqueeze(0).expand(self.opt.batch_size, 2, self.opt.img_H,
                                                         self.opt.img_W)  # Bx2xHxW
        '''
    def forward(self, x):
        #K(B,3,3)
        
        resnet_out = self.backbone(x)
        return resnet_out



class ResidualConv(nn.Module):
    def __init__(self,inplanes,planes,stride=1,kernel_1=False):
        super(ResidualConv,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if kernel_1:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,bias=False),
                nn.BatchNorm2d(planes))
        else:

            self.conv_skip = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1,bias=False),
                nn.BatchNorm2d(planes))

        self.stride = stride

    def forward(self, x):
        identity = self.conv_skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class attention_pc2img(nn.Module):
    def __init__(self,in_channel,output_channel):
        super(attention_pc2img,self).__init__()
        '''self.conv=nn.Sequential(nn.Conv2d(in_channel,in_channel,1),nn.BatchNorm2d(in_channel),nn.ReLU(),
                                nn.Conv2d(in_channel,in_channel,1),nn.BatchNorm2d(in_channel),nn.ReLU(),
                                nn.Conv2d(in_channel,output_channel,1),nn.BatchNorm2d(output_channel),nn.ReLU())'''
        self.conv=nn.Sequential(ResidualConv(in_channel,in_channel),ResidualConv(in_channel,in_channel),nn.Conv2d(in_channel,output_channel,1),nn.BatchNorm2d(output_channel),nn.ReLU())
    def forward(self,pc_global_feature,img_local_feature,pc_local_feature):
        #print(img_local_feature.size(),pc_global_feature.size())
        B,_,H,W=img_local_feature.size()
        feature=torch.cat([img_local_feature,pc_global_feature.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)],dim=1)
        feature=self.conv(feature)
        attention=F.softmax(feature,dim=1)
        #print(attention.size())
        #print(pc_local_feature.size())
        feature_fusion=torch.sum(attention.unsqueeze(1)*pc_local_feature.unsqueeze(-1).unsqueeze(-1),dim=2)
        return feature_fusion

class ImageUpSample(nn.Module):
    def __init__(self,in_channel,output_channel):
        super(ImageUpSample,self).__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        #self.up=nn.ConvTranspose2d(in_channel,in_channel,kernel_size=3,stride=2)
        self.conv=nn.Sequential(ResidualConv(in_channel,output_channel),ResidualConv(output_channel,output_channel))
        '''self.conv=nn.Sequential(nn.Conv2d(in_channel,output_channel,1,bias=False),nn.BatchNorm2d(output_channel),nn.ReLU(),
                                nn.Conv2d(output_channel,output_channel,1,bias=False),nn.BatchNorm2d(output_channel),nn.ReLU(),
                                nn.Conv2d(output_channel,output_channel,1,bias=False),nn.BatchNorm2d(output_channel),nn.ReLU())'''
    def forward(self,x1,x2):
        #x1: downsampled
        x1=self.up(x1)
        x=self.conv(torch.cat((x1,x2),dim=1))
        return x


class Image_ResNet(nn.Module):
    def __init__(self):
        super(Image_ResNet, self).__init__()
        
        self.encoder = ImageEncoder()
        
        self.up_conv1=ImageUpSample(768,256)
        self.up_conv2=ImageUpSample(256+128,128)
        self.up_conv3=ImageUpSample(128+64+64,64)

        # image mesh grid
        '''input_mesh_np = np.meshgrid(np.linspace(start=0, stop=self.opt.img_W - 1, num=self.opt.img_W),
                                    np.linspace(start=0, stop=self.opt.img_H - 1, num=self.opt.img_H))
        input_mesh = torch.from_numpy(np.stack(input_mesh_np, axis=0).astype(np.float32)).to(self.opt.device)  # 2xHxW
        self.input_mesh = input_mesh.unsqueeze(0).expand(self.opt.batch_size, 2, self.opt.img_H,
                                                         self.opt.img_W)  # Bx2xHxW
        '''
    def forward(self, x):
        #K(B,3,3)
        img_feature_set=self.encoder(x)

        img_global_feature=img_feature_set[-1]  #512
        img_s32_feature_map=img_feature_set[-2] #512
        img_s16_feature_map=img_feature_set[-3] #256
        img_s8_feature_map=img_feature_set[-4]  #128
        img_s4_feature_map=img_feature_set[-5]  #64
        img_s2_feature_map=img_feature_set[-6]  #64

        image_feature_16=self.up_conv1(img_s32_feature_map,img_s16_feature_map)
        image_feature_8=self.up_conv2(image_feature_16,img_s8_feature_map)
        img_s4_feature_map=torch.cat((img_s4_feature_map,F.interpolate(img_s2_feature_map,scale_factor=0.5)),dim=1)
        image_feature_mid=self.up_conv3(image_feature_8,img_s4_feature_map)

        return img_global_feature.squeeze(-1).squeeze(-1), image_feature_mid

if __name__=='__main__':
    a=torch.rand(10,3,160,512).cuda()
    model=Image_ResNet()
    model=model.cuda()
    b, c, d=model(a)
    print(b.shape, c.shape, d.shape)