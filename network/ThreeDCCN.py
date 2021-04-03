import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import script.common as cm


class BaseNet(nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs):
        res = self.forward_one(imgs)
        return res


class Cyclindrical_ConvNet(BaseNet):
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn_2d(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _make_bn_3d(self, outd):
        return nn.BatchNorm3d(outd, affine=self.bn_affine)

    def _add_conv_2d(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        self.dilation *= stride
        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=(k, k), dilation=d))
        if bn and self.bn: self.ops.append(self._make_bn_2d(outd))
        if relu: self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def _add_conv_3d(self, outd, k, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        self.dilation *= stride
        self.ops.append(nn.Conv3d(self.curchan, outd, kernel_size=(k[0], k[1], k[2]), dilation=d))
        if bn and self.bn: self.ops.append(self._make_bn_3d(outd))
        if relu: self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            k_exist = hasattr(op, 'kernel_size')
            if k_exist:
                if len(op.kernel_size) == 3:
                    x = cm.pad_image_3d(x, op.kernel_size[1] + (op.kernel_size[1] - 1) * (op.dilation[0] - 1))
                else:
                    if len(x.shape) == 5:
                        x = x.squeeze(2)
                    x = cm.pad_image(x, op.kernel_size[0] + (op.kernel_size[0] - 1) * (op.dilation[0] - 1))
            x = op(x)
        return x


class Cylindrical_Net(Cyclindrical_ConvNet):
    """ Compute a descriptor for all overlapping patches.
        From the L2Net paper (CVPR'17).
    """

    def __init__(self, inchan=16, dim=32, **kw):
        Cyclindrical_ConvNet.__init__(self, inchan=inchan, **kw)
        add_conv_2d = lambda n, **kw: self._add_conv_2d(n, **kw)
        add_conv_3d = lambda n, **kw: self._add_conv_3d(n, **kw)
        add_conv_3d(32, k=[3, 3, 3])
        add_conv_3d(32, k=[3, 3, 3])
        add_conv_3d(64, k=[3, 3, 3])
        add_conv_3d(64, k=[3, 3, 3])
        add_conv_2d(128, stride=2)
        add_conv_2d(128)
        add_conv_2d(64, stride=2)
        add_conv_2d(64)
        add_conv_2d(32, k=2, stride=2, relu=False)
        add_conv_2d(32, k=2, stride=2, relu=False)
        add_conv_2d(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim
