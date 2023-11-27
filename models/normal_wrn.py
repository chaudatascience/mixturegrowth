## PAPER: Wide Residual Networks (2017)- https://arxiv.org/pdf/1605.07146.pdf
## Official code: https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py

###### WRN-n-k
# n: total number of convolutional layers
# k: widening factor

###### Comparisons between ResNet and WRN (from the paper)
# WRN is the same as ResNet, except
# 1/ the number of channels in each layer is increased by a factor of k:
# e.g., in resnet:     group_dims = [64, 128, 256, 512], with 4 groups of blocks
# for Wide resnet:     group_dims = [16, 16 * width, 32 * width, 64 * width], with 3 (one fewer) groups of blocks
# 2/ The order of conv and batch norm, relu: From conv-BN-ReLU (in Resnet) to BN-ReLU-conv (in WRN)
# (page 4)

###### Architecture of WRN
## 1 conv stem -> 3 groups -> fc
# zoom in the "groups" part:
# each group has n blocks, each block has b conv layers (b=2 for BasicBlock, 3 for Bottleneck, but in WRN, we focus on BasicBlock)
# note that each group may also have a shortcut at the first block, so there are  n*b + 1 conv layers in each group
# so, in total, the number of conv layers are
# 1 + 3 * (n*b + 1) = 3n*b + 4
# with b=2 (BasicBlock), we have 6n + 4 conv layers in total.

###### Example: WRN-28-10: 28 conv layers, 10x widening factor
# 1 conv stem -> 3 groups -> 1 FC.

# Each group consists of 4 blocks and  1 shortcut (optional).
# Each block has 2 conv layers (denoted as B(3,3) in the paper: two 3x3 conv layers)
# so, in total: 1 + 3 * (4*2 + 1) = 28 conv layers

import torch
from torch import nn, Tensor
from torch.nn import init
import utils
from compute_flops import compute_flops



def conv1x1(in_dim: int, out_dim: int, stride: int = 1) -> nn.Conv2d:
    """return 1x1 conv"""
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_dim: int, out_dim: int, stride: int, padding: int) -> nn.Conv2d:
    """return 3x3 conv"""
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride, norm):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = conv3x3(in_dim, out_dim, stride=stride,
                             padding=1)  # stride = `stride` for the first conv

        if norm:
            self.bn2 = nn.BatchNorm2d(out_dim)
        else:
            self.bn2 = nn.Identity()

        self.conv2 = conv3x3(out_dim, out_dim, stride=1, padding=1)  # stride = 1 for the second conv

        self.relu = nn.ReLU(inplace=True)
        self.in_out_equal = (in_dim == out_dim)
        self.shortcut = None
        if not self.in_out_equal:
            self.shortcut = conv1x1(in_dim, out_dim, stride=stride)


    def forward(self, x: Tensor) -> Tensor:
        ## main branch: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))

        ## residual branch: BN -> ReLU -> Conv1x1
        if not self.in_out_equal:
            residual = self.shortcut(self.relu(self.bn1(x)))
        else:
            residual = x

        return out + residual



class NormalWRN(nn.Module):
    def __init__(self, input_dim: int, depth: int, width: float, num_classes: int, norm: bool):
        """
        Paper: Wide Residual Networks - https://arxiv.org/pdf/1605.07146.pdf
        """
        super().__init__()

        group_dims = [16, int(16 * width), int(32 * width), int(64 * width)]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        num_blocks = (depth - 4) // 6  # num blocks for each group

        self.num_classes = num_classes
        self.conv_stem = conv3x3(input_dim, group_dims[0], stride=1, padding=1)

        self.group_1 = self._make_group(group_dims[0], group_dims[1], num_blocks, 1, norm)
        self.group_2 = self._make_group(group_dims[1], group_dims[2], num_blocks, 2, norm)
        self.group_3 = self._make_group(group_dims[2], group_dims[3], num_blocks, 2, norm)

        self.last_act = nn.Sequential(
            nn.BatchNorm2d(group_dims[3]),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(group_dims[3], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_group(self, in_dim: int, out_dim: int, num_blocks: int, first_stride: int = 1, norm=False) -> nn.Sequential:
        blocks = []
        blocks.append(BasicBlock(in_dim, out_dim, first_stride, norm=norm))
        for i in range(1, num_blocks):
            blocks.append(BasicBlock(out_dim, out_dim, 1, norm=norm))
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor, member=None) -> Tensor:
        """
        member_id is ignored in this model
        """
        assert member is None, 'member_id should be None!'
        out = self.conv_stem(x)
        out = self.group_1(out)
        out = self.group_2(out)
        out = self.group_3(out)
        out = self.last_act(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


def normalwrn(num_classes: int, args) -> NormalWRN:
    input_dim, depth, width, norm = 3, args.depth, args.wide, args.use_bn
    model = NormalWRN(input_dim, depth, width, num_classes, norm=norm)
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--wide', type=int, default=.5)
    parser.add_argument('--use_bn', action='store_true')
    args = parser.parse_args()

    model = normalwrn(100, args)
    x = torch.randn(1, 3, 32, 32)
    y = model(x, member=None)

    utils.analyze_model(model)
    flops = compute_flops(model, x)
    print(flops/1e9)
    ## WRN 28-1
    # Total parameters: 374,452;	Trainable: 374,452
    # 0.111624704

    ## WRN 28-0.25
    # Total parameters: 25,796;	Trainable: 25,796
    # 0.009425024