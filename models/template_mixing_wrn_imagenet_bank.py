import torch
import torch.nn as nn
from torch.nn import init

from compute_flops import compute_flops, compute_flops_with_members
from models.layers import *
from models.bank_cfg import bank_cfg

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


if __name__ == "__main__":
    args_imagenet = {
        "data_path": "/projectnb/ivc-ml/dbash/data/imagenet/ILSVRC/Data/CLS-LOC",
        "dataset": "imagenet",
        "arch": "wrn_imagenet",
        "effnet_arch": None,
        "depth": 56,
        "wide": 4.0,
        "member_template_sets": [0, 1, 0, 1],
        "num_templates_each_set": [2, 2],
        "log_model": True,
        "member_ids": [],
        "growth_epochs": [90],
        "template_size": [0.5, 0.5, 1.0],
        "ensemble_growth_type": "diag",
        "ensemble_train_epochs": 90,
        "lr_cosine_epoch": "reset",
        "warmup_member_0": 0,
        "log_warmup_member_0": False,
        "member_0_loss_weight": 1,
        "switch_training": True,
        "resume_member_0": "snapshots_v2/imagenet/two_templ_share/seed_957472/checkpoint_epoch90.pth.tar",
        "resume_member_1": "snapshots_v2/imagenet/switchv8/seed_541480/checkpoint_epoch90.pth.tar",
        "scale_templates_and_coefs": False,
        "copy_templates_and_bn_0_to_1": False,
        "copy_templates_and_bn_0_to_1_noise": 0.01,
        "reset_scheduler": True,
        "reset_optimizer": False,
        "lr_schedule_as_member_0": True,
        "coefficient_init_type": "orthogonal",
        "cross_layer_sharing": False,
        "lr_growing": 0.007598,
        "small_lr": 1e-05,
        "lr_growing_min": 1e-06,
        "reset_lr_scheduler_growing": "1",
        "reset_optimizer_growing": True,
        "coefs_growing": "none",
        "coefs_noise_growing": 1.0,
        "warmup_01_growing": 0,
        "scale_templates_and_coefs_growing": False,
        "lr_warmup_growing": 0,
        "add_extra_templates_growing": 0,
        "epochs": 196,
        "batch_size": 256,
        "eval_batch_size": 512,
        "drop_last": False,
        "learning_rate": 0.007598,
        "momentum": 0.9,
        "no_nesterov": True,
        "label_smoothing": 0.0,
        "optimizer": "sgd",
        "scheduler_type": "cosine",
        "schedule": None,
        "gammas": None,
        "warmup_epochs": None,
        "base_lr": 0.1,
        "step_size": None,
        "step_gamma": None,
        "step_warmup": None,
        "decay": 0.0001,
        "use_bn": False,
        "no_bn_decay": False,
        "cutout": False,
        "ema_decay": None,
        "print_freq": 100,
        "save_path": "./snapshots_v2/imagenet/test2/seed_300799",
        "resume": "",
        "start_epoch": 180,
        "evaluate": False,
        "no_cifar_full": False,
        "best_loss": False,
        "ngpu": 2,
        "workers": 6,
        "dist": False,
        "amp": False,
        "no_dp": False,
        "no_wandb": False,
        "wandb_log_freq": 7800,
        "manualSeed": 300799,
        "tag": "test2",
        "scc_id": "7393238",
        "debug": False,
        "auto_chose_batch_size": True,
        "use_cuda": True,
    }

    def update_children(model, ind, evaluation):
        for layer in model.children():
            if isinstance(layer, (SConv2d, SLinear, SBatchNorm2d)):
                layer.growth_update(ind, evaluation)
            else:
                update_children(layer, ind, evaluation)

    def growth_update(model, ind=None, evaluation=False, args=None):
        if ind is None:
            # assumes we are testing the final model
            ind = len(args.template_size) - 1

        update_children(model, ind, evaluation)

    from utils import analyze_model

    ## IMAGENET
    print("IMAGENET")
    args_imagenet = utils.dict2obj(args_imagenet)

    model = wrn_imagenet_bank(num_classes=1000, args=args_imagenet)
    # save `model` as a dictionary
    model_dict = model.state_dict()

    # save `model_dict` as a dictionary
    # torch.save(model_dict, 'model_dict.pt')

    x = torch.randn(1, 3, 224, 224)
    members = [0]
    flops_student_0 = compute_flops_with_members(model, x, member_id=members)
    print("flops_student_0: ", flops_student_0)
    print(model(x, 1).shape)

    #
    #
    members = None
    growth_update(model, args=args_imagenet)
    flops = compute_flops_with_members(model, x, member_id=members)
    print("flops: ", flops)
    # total_params, trainable_params = analyze_model(model, False)
    # print("total_params: ", total_params)
    # print("trainable_params: ", trainable_params)
    # #
    # print("------------------------")
    #
