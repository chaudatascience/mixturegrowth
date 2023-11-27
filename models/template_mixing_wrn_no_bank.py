import torch
import torch.nn as nn
from torch.nn import init

import utils
from compute_flops import compute_flops, compute_flops_with_members
from .layers import *


class Identity2(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x, member_id=None):
        return self.identity(x)


class Block(nn.Module):
    def __init__(self, args, in_planes, out_planes, stride, norm):
        super(Block, self).__init__()
        self.bn1 = SBatchNorm2d(args, in_planes)
        if norm:
            self.bn2 = SBatchNorm2d(args, out_planes)
        else:
            self.bn2 = Identity2()

        self.relu = nn.ReLU(inplace=True)
        self.equalInOut = in_planes == out_planes
        self.convShortcut = None

        if args.cross_layer_sharing and self.equalInOut:
            num_shared_layers = 2
            shared_layer = SConv2d(
                args,
                num_shared_layers,
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
            self.conv1 = shared_layer
            self.conv2 = shared_layer
        else:
            num_shared_layers = 1
            self.conv1 = SConv2d(
                args,
                num_shared_layers,
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
            self.conv2 = SConv2d(
                args,
                num_shared_layers,
                out_planes,
                out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            if not self.equalInOut:
                self.convShortcut = SConv2d(
                    args,
                    num_shared_layers,
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                )

    def forward(self, x, member_id):
        residual = x
        out = self.relu(self.bn1(x, member_id))
        if not self.equalInOut:
            residual = out
        out = self.conv2(
            self.relu(self.bn2(self.conv1(out, member_id), member_id)), member_id
        )
        if self.convShortcut is not None:
            residual = self.convShortcut(residual, member_id)
        return out + residual


class WRN(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        width = args.wide
        depth = args.depth
        norm = args.use_bn
        n_channels = [16, int(16 * width), int(32 * width), int(64 * width)]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        print("WRN : Depth : {} , Widen Factor : {}".format(depth, width))

        num_shared_layers = 1
        self.conv_3x3 = SConv2d(
            args,
            num_shared_layers,
            3,
            n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            set_input=True,
        )

        self.stage_1 = self._make_layer(
            args, n_channels[0], n_channels[1], num_blocks, 1, norm=norm
        )
        self.stage_2 = self._make_layer(
            args, n_channels[1], n_channels[2], num_blocks, 2, norm=norm
        )
        self.stage_3 = self._make_layer(
            args, n_channels[2], n_channels[3], num_blocks, 2, norm=norm
        )

        self.last_bn = SBatchNorm2d(args, n_channels[3])
        self.lastact = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = SLinear(args, num_shared_layers, n_channels[3], num_classes)

    def _make_layer(
        self, args, in_planes, out_planes, num_blocks, stride=1, norm=False
    ):
        blocks = []
        blocks.append(Block(args, in_planes, out_planes, stride, norm=norm))
        for i in range(1, num_blocks):
            blocks.append(Block(args, out_planes, out_planes, 1, norm=norm))
        return Sequential(*blocks)

    def forward(self, x, member_id=None):
        x = self.conv_3x3(x, member_id)
        x = self.stage_1(x, member_id)
        x = self.stage_2(x, member_id)
        x = self.stage_3(x, member_id)
        x = self.lastact(self.last_bn(x, member_id))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x, member_id)


def wrn(num_classes, args):
    model = WRN(args, num_classes)
    return model


# ImageNet model.
# This is a ResNet v1.5-style model (stride 2 on 3x3 convolutions).
# In contrast to the above, this applies batchnorm/relu after convolution.


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
        self, args, in_channels, mid_channels, downsample, width=1, pool_residual=False
    ):
        super().__init__()
        self.out_channels = 4 * mid_channels
        # Width factor applies only to inner 3x3 convolution.
        mid_channels = int(mid_channels * width)

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
                )
        elif in_channels != self.out_channels:
            self.skip_connection = ConvBNRelu(
                args,
                in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                relu=False,
            )
        else:
            self.skip_connection = None

        # Main branch.
        self.in_conv = ConvBNRelu(
            args, in_channels, mid_channels, kernel_size=1, stride=1, padding=0
        )
        self.mid_conv = ConvBNRelu(
            args,
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=(2 if downsample else 1),
            padding=1,
        )
        self.out_conv = ConvBNRelu(
            args,
            mid_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            relu=False,
        )
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


class ResNet(nn.Module):
    def __init__(
        self,
        args,
        block,
        module_sizes,
        module_channels,
        num_classes,
        width=1,
        pool_residual=False,
    ):
        super().__init__()

        num_shared_layers = 1
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
        )
        self.conv2 = ConvBNRelu(
            args,
            module_channels[0] // 2,
            module_channels[0] // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            num_shared_layers=num_shared_layers,
        )
        self.conv3 = ConvBNRelu(
            args,
            module_channels[0] // 2,
            module_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            num_shared_layers=num_shared_layers,
        )
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
                )
                out_channels = b.out_channels
                blocks.append(b)
            modules.append(Sequential(*blocks))
        self.block_modules = Sequential(*modules)

        # Output.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SLinear(args, num_shared_layers, out_channels, num_classes)

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


def wrn_imagenet(num_classes, args):
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
    model = ResNet(
        args,
        BottleneckBlock,
        module_sizes=(3, 4, 6, 3),
        module_channels=channels,
        num_classes=num_classes,
        width=mid_width,
        pool_residual=False,
    )

    return model


# if __name__ == '__main__':
#
#     args_imagenet= {'data_path': '/projectnb/ivc-ml/dbash/data/imagenet/ILSVRC/Data/CLS-LOC', 'dataset': 'imagenet', 'arch': 'wrn_imagenet', 'effnet_arch': None, 'depth': 56, 'wide': 4, 'num_templates': 4, 'growth_epochs': [100], 'template_size': [0.5, 0.5, 1.0], 'ensemble_growth_type': 'diag', 'ensemble_train_epochs': 22, 'switch_training': False, 'resume_imagenet': True, 'scale_templates_and_coefs': False, 'scale_templates_and_coefs_growing': False, 'copy_templates_0_to_1': False, 'copy_templates_0_to_1_growing': False, 'reset_scheduler': False, 'reset_scheduler_when_growing': False, 'post_growth_templates': 0, 'model_0_templates': [0, 1], 'model_1_templates': [2, 3], 'model_2_templates': [], 'model_3_templates': [], 'coefficient_init_type': 'orthogonal', 'cross_layer_sharing': False, 'epochs': 172, 'batch_size': 512, 'drop_last': False, 'learning_rate': 0.1, 'momentum': 0.9, 'no_nesterov': True, 'label_smoothing': 0.0, 'optimizer': 'sgd', 'scheduler_type': 'cosine', 'schedule': None, 'gammas': None, 'warmup_epochs': None, 'base_lr': 0.1, 'step_size': None, 'step_gamma': None, 'step_warmup': None, 'decay': 0.0001, 'use_bn': False, 'no_bn_decay': False, 'cutout': False, 'ema_decay': None, 'print_freq': 100, 'save_path': './snapshots_v2/imagenet/ours/seed_769345', 'resume': '/projectnb/ivc-ml/chaupham/growing_v2/snapshots_v2/imagenet/ours/checkpoint61.pth.tar', 'start_epoch': 0, 'evaluate': False, 'cifar_full': True, 'best_loss': False, 'ngpu': 4, 'workers': 14, 'dist': False, 'amp': False, 'no_dp': False, 'manualSeed': 769345, 'tag': 'ours', 'debug': False, 'use_cuda': True}
#
#     args_cifar100 = {'data_path': 'data', 'dataset': 'cifar100', 'arch': 'wrn', 'effnet_arch': None, 'depth': 28, 'wide': 10, 'num_templates': 4, 'growth_epochs': [100], 'template_size': [0.5, 0.5, 1.0], 'ensemble_growth_type': 'diag', 'ensemble_train_epochs': 100, 'switch_training': True, 'resume_model_0': '', 'scale_templates_and_coefs': True, 'scale_templates_and_coefs_growing': True, 'copy_templates_0_to_1': False, 'copy_templates_0_to_1_growing': False, 'copy_templates_0_to_1_noise': 0.1, 'reset_scheduler': True, 'reset_scheduler_when_growing': False, 'post_growth_templates': 0, 'model_0_templates': [0, 1], 'model_1_templates': [2, 3], 'model_2_templates': [0, 1, 2, 3], 'model_3_templates': [0, 1, 2, 3], 'coefficient_init_type': 'orthogonal', 'cross_layer_sharing': False, 'epochs': 220, 'batch_size': 128, 'drop_last': False, 'learning_rate': 0.1, 'momentum': 0.9, 'no_nesterov': False, 'label_smoothing': 0.0, 'optimizer': 'sgd', 'scheduler_type': 'cosine', 'schedule': [30, 60, 90], 'gammas': [0.2, 0.2, 0.2], 'warmup_epochs': None, 'base_lr': 0.1, 'step_size': None, 'step_gamma': None, 'step_warmup': None, 'decay': 0.0005, 'use_bn': False, 'no_bn_decay': False, 'cutout': True, 'ema_decay': None, 'lr_growing': 0.001, 'print_freq': 100, 'save_path': './snapshots_v2/cifar100/ours/seed_402640', 'resume': 'snapshots_v2/cifar100/ours/seed_559246/checkpoint_epoch100.pth.tar', 'start_epoch': 0, 'evaluate': False, 'cifar_full': True, 'best_loss': False, 'ngpu': 1, 'workers': 2, 'dist': False, 'amp': False, 'no_dp': False, 'no_wandb': False, 'wandb_log_freq': 7800, 'manualSeed': 402640, 'tag': 'ours', 'scc_id': '796056', 'debug': False, 'use_cuda': True}
#
#     def update_children(model, ind, evaluation):
#         for layer in model.children():
#             if isinstance(layer, (SConv2d, SLinear, SBatchNorm2d)):
#                 layer.growth_update(ind, evaluation)
#             else:
#                 update_children(layer, ind, evaluation)
#
#
#     def growth_update(model, ind=None, evaluation=False, args=None):
#         if ind is None:
#             # assumes we are testing the final model
#             ind = len(args.template_size) - 1
#
#         update_children(model, ind, evaluation)
#
#     from utils import analyze_model
#
#     ## IMAGENET
#     print("IMAGENET")
#     args_imagenet = utils.dict2obj(args_imagenet)
#
#     model = wrn_imagenet(num_classes=1000, args=args_imagenet)
#     x = torch.randn(1, 3, 224, 224)
#     members = [0]
#     flops_student_0 = compute_flops_with_members(model, x, member_id=members)
#     print("flops_student_0: ", flops_student_0)
#
#     members = None
#     growth_update(model, args=args_imagenet)
#     flops = compute_flops_with_members(model, x, member_id=members)
#     print("flops: ", flops)
#     total_params, trainable_params = analyze_model(model, False)
#     print("total_params: ", total_params)
#     print("trainable_params: ", trainable_params)
#
#     print("------------------------")
#
#
#
#     ## CIFAR100
#     print("CIFAR100")
#     args_cifar100 = utils.dict2obj(args_cifar100)
#     model = wrn(num_classes=100, args=args_cifar100)
#     x = torch.randn(1, 3, 32, 32)
#     members = [0]
#     flops_student_0 = compute_flops_with_members(model, x, member_id=members)
#     print("flops_student_0: ", flops_student_0)
#
#     members = None
#     growth_update(model, args=args_cifar100)
#     flops = compute_flops_with_members(model, x, member_id=members)
#     print("flops: ", flops)
#     total_params, trainable_params = analyze_model(model, False)
#     print("total_params: ", total_params)
#     print("trainable_params: ", trainable_params)
#     print("------------------------")
