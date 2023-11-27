import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, Sequential, init, Linear

from compute_flops import compute_flops
from utils import analyze_model


class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, stride, kernel_size, padding, relu=True):
        super().__init__()

        self.conv = Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn = BatchNorm2d(out_planes)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.relu is not None:
            out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample, width=1, pool_residual=False):
        super().__init__()
        self.out_channels = 4 * mid_channels
        # Width factor applies only to inner 3x3 convolution.
        mid_channels = int(mid_channels * width)

        # Skip connection.
        if downsample:
            if pool_residual:
                pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                conv = ConvBNRelu(
                    in_channels, self.out_channels, stride=1, kernel_size=1, padding=0, relu=False
                )
                self.skip_connection = Sequential(pool, conv)
            else:
                self.skip_connection = ConvBNRelu(
                    in_channels, self.out_channels, kernel_size=1, stride=2, padding=0, relu=False
                )
        elif in_channels != self.out_channels:
            self.skip_connection = ConvBNRelu(
                in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, relu=False
            )
        else:
            self.skip_connection = None

        # Main branch.
        self.in_conv = ConvBNRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.mid_conv = ConvBNRelu(
            mid_channels, mid_channels, kernel_size=3, stride=(2 if downsample else 1), padding=1
        )
        self.out_conv = ConvBNRelu(
            mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, relu=False
        )
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.skip_connection is not None:
            residual = self.skip_connection(x)
        else:
            residual = x

        out = self.out_conv(self.mid_conv(self.in_conv(x)))
        out += residual
        return self.out_relu(out)


class ResNet(nn.Module):
    def __init__(
        self, block, module_sizes, module_channels, num_classes, width=1, pool_residual=False
    ):
        super().__init__()

        # Input trunk, Inception-style.
        self.conv1 = ConvBNRelu(3, module_channels[0] // 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNRelu(
            module_channels[0] // 2, module_channels[0] // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = ConvBNRelu(
            module_channels[0] // 2, module_channels[0], kernel_size=3, stride=1, padding=1
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the main network.
        modules = []
        out_channels = module_channels[0]
        for module_idx, (num_layers, mid_channels) in enumerate(zip(module_sizes, module_channels)):
            blocks = []
            for i in range(num_layers):
                in_channels = out_channels
                downsample = i == 0 and module_idx > 0
                b = block(
                    in_channels, mid_channels, downsample, width=width, pool_residual=pool_residual
                )
                out_channels = b.out_channels
                blocks.append(b)
            modules.append(Sequential(*blocks))
        self.block_modules = Sequential(*modules)

        # Output.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(out_channels, num_classes)

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

    def forward(self, x, member=None):
        assert member is None
        x = self.maxpool(self.conv3(self.conv2(self.conv1(x))))
        x = self.block_modules(x)
        x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x


def normal_wrn_imagenet(num_classes, args):
    """ResNet-50, with optional width (depth ignored for now, can generalize)."""
    assert num_classes == 1000
    width = args.wide
    mid_width = 1
    channels = (
        int(64 / 4 * width),
        int(128 / 4 * width),
        int(256 / 4 * width),
        int(512 / 4 * width),
    )
    model = ResNet(
        BottleneckBlock,
        module_sizes=(3, 4, 6, 3),
        module_channels=channels,
        num_classes=num_classes,
        width=mid_width,
        pool_residual=False,
    )

    return model
