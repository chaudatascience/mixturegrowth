## In this implementation, we use a parameter bank ("bank_cfg" and "bank") to share parameters across layers
# ImageNet model.
# This is a ResNet v1.5-style model (stride 2 on 3x3 convolutions).
# In contrast to the above, this applies batchnorm/relu after convolution.


import torch
import torch.nn as nn
from torch.nn import init

from compute_flops import compute_flops, compute_flops_with_members
from models.layers import *
from models.bank_cfg import bank_cfg

class ConvBNRelu(nn.Module):
    def __init__(
        self,
        args,
        in_planes,
        out_planes,
        stride,
        kernel_size,
        padding,
        relu=True,
        num_shared_layers=1,
        set_input=False,
        bank_cfg=None,
        bank=None,
        layer_id=None,
    ):
        super().__init__()

        self.conv = SConv2d(
            args,
            num_shared_layers,
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            set_input=set_input,
            bank_cfg=bank_cfg,
            bank=bank,
            layer_id=layer_id,
        )

        self.bn = SBatchNorm2d(args, out_planes)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x, member_id):
        out = self.bn(self.conv(x, member_id), member_id)
        if self.relu is not None:
            out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        args,
        in_channels,
        mid_channels,
        downsample,
        width=1,
        pool_residual=False,
        layer_id=None,
        bank_cfg=None,
        bank=None,
    ):
        super().__init__()
        self.out_channels = 4 * mid_channels
        # Width factor applies only to inner 3x3 convolution.
        mid_channels = int(mid_channels * width)

        self.layer_id = layer_id
        # Skip connection.
        if downsample:
            if pool_residual:
                pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                conv = ConvBNRelu(
                    args,
                    in_channels,
                    self.out_channels,
                    stride=1,
                    kernel_size=1,
                    padding=0,
                    relu=False,
                    bank_cfg=bank_cfg,
                    bank=bank,
                    layer_id=self.layer_id,
                )
                self.skip_connection = Sequential(pool, conv)
            else:
                self.skip_connection = ConvBNRelu(
                    args,
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    relu=False,
                    bank_cfg=bank_cfg,
                    bank=bank,
                    layer_id=self.layer_id,
                )
            self.layer_id += 1
        elif in_channels != self.out_channels:
            self.skip_connection = ConvBNRelu(
                args,
                in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=False,
                bank_cfg=bank_cfg,
                bank=bank,
                layer_id=self.layer_id,
            )
            self.layer_id += 1
        else:
            self.skip_connection = None

        # Main branch.
        self.in_conv = ConvBNRelu(
            args,
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bank_cfg=bank_cfg,
            bank=bank,
            layer_id=self.layer_id,
        )
        self.layer_id += 1
        self.mid_conv = ConvBNRelu(
            args,
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=(2 if downsample else 1),
            padding=1,
            bank_cfg=bank_cfg,
            bank=bank,
            layer_id=self.layer_id,
        )
        self.layer_id += 1
        self.out_conv = ConvBNRelu(
            args,
            mid_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            relu=False,
            bank_cfg=bank_cfg,
            bank=bank,
            layer_id=self.layer_id,
        )
        self.layer_id += 1
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x, member_id):
        if self.skip_connection is not None:
            residual = self.skip_connection(x, member_id)
        else:
            residual = x

        out = self.out_conv(
            self.mid_conv(self.in_conv(x, member_id), member_id), member_id
        )
        out += residual
        return self.out_relu(out)


class ResNetBank(nn.Module):
    def __init__(
        self,
        args,
        block,
        module_sizes,
        module_channels,
        num_classes,
        width=1,
        pool_residual=False,
        bank_cfg=None,
    ):
        super().__init__()

        num_shared_layers = 1
        self.bank = nn.ParameterDict()
        self.layer_id = 0

        # Input trunk, Inception-style.
        self.conv1 = ConvBNRelu(
            args,
            3,
            module_channels[0] // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            num_shared_layers=num_shared_layers,
            set_input=True,
            bank_cfg=bank_cfg,
            bank=self.bank,
            layer_id=self.layer_id,
        )
        self.layer_id += 1

        self.conv2 = ConvBNRelu(
            args,
            module_channels[0] // 2,
            module_channels[0] // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            num_shared_layers=num_shared_layers,
            bank_cfg=bank_cfg,
            bank=self.bank,
            layer_id=self.layer_id,
        )
        self.layer_id += 1

        self.conv3 = ConvBNRelu(
            args,
            module_channels[0] // 2,
            module_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            num_shared_layers=num_shared_layers,
            bank_cfg=bank_cfg,
            bank=self.bank,
            layer_id=self.layer_id,
        )
        self.layer_id += 1

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the main network.
        modules = []
        out_channels = module_channels[0]
        for module_idx, (num_layers, mid_channels) in enumerate(
            zip(module_sizes, module_channels)
        ):
            blocks = []
            for i in range(num_layers):
                in_channels = out_channels
                downsample = i == 0 and module_idx > 0
                b = block(
                    args,
                    in_channels,
                    mid_channels,
                    downsample,
                    width=width,
                    pool_residual=pool_residual,
                    bank_cfg=bank_cfg,
                    bank=self.bank,
                    layer_id=self.layer_id,
                )
                self.layer_id = b.layer_id
                out_channels = b.out_channels
                blocks.append(b)
            modules.append(Sequential(*blocks))
        self.block_modules = Sequential(*modules)

        # Output.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SLinear(
            args,
            num_shared_layers,
            out_channels,
            num_classes,
            bank_cfg=bank_cfg,
            bank=self.bank,
            layer_id=self.layer_id,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x, member_id):
        x = self.maxpool(
            self.conv3(self.conv2(self.conv1(x, member_id), member_id), member_id)
        )
        x = self.block_modules(x, member_id)
        x = self.fc(torch.flatten(self.avgpool(x), 1), member_id)
        return x


def wrn_imagenet_bank(num_classes, args):
    """ResNet-50, (wrn56-4) with optional width (depth ignored for now, can generalize)."""
    assert num_classes == 1000
    width = args.wide
    mid_width = 1
    channels = (
        int(64 / 4 * width),
        int(128 / 4 * width),
        int(256 / 4 * width),
        int(512 / 4 * width),
    )
    model = ResNetBank(
        args,
        BottleneckBlock,
        module_sizes=(3, 4, 6, 3),
        module_channels=channels,
        num_classes=num_classes,
        width=mid_width,
        pool_residual=False,
        bank_cfg=bank_cfg,
    )

    return model