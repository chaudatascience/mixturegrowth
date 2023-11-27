from __future__ import division
import os, sys, shutil, time, random
import argparse
import warnings
import contextlib
import copy
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from timm.scheduler import CosineLRScheduler

import utils
from models.net2net import net2net_wrn28_k_wider, net2net_wrn_imagenet_wider
from compute_flops import compute_flops_with_members
from utils import (
    AverageMeter,
    RecorderMeter,
    time_string,
    convert_secs2time,
    Cutout,
    Lighting,
    LabelSmoothingNLLLoss,
    RandomDataset,
    PrefetchWrapper,
    fast_collate,
    get_world_rank,
    get_world_size,
    get_local_rank,
    initialize_dist,
    get_cuda_device,
    allreduce_tensor,
    datetime_now,
    analyze_model,
    type_list,
)
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import custom_log
import models
import random
import PIL
from models.layers import *
from models import layers

# Ignore corrupted TIFF warnings in ImageNet.
warnings.filterwarnings("ignore", message=".*(C|c)orrupt\sEXIF\sdata.*")
# Ignore anomalous warnings from learning rate schedulers with GradScaler.
warnings.filterwarnings("ignore", message=".*lr_cosine_scheduler\.step.*")

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(
    description="Training script",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Data / Model
parser.add_argument("data_path", metavar="DPATH", type=str, help="Path to dataset")
parser.add_argument(
    "--dataset",
    metavar="DSET",
    type=str,
    choices=["cifar10", "cifar100", "imagenet", "rand_imagenet"],
    help="Choose between CIFAR/ImageNet.",
)

parser.add_argument(
    "--arch",
    metavar="ARCH",
    default="wrn",
    choices=[
        "wrn",
        "normalwrn",
        "wrn_imagenet",
        "wrn_imagenet_bank",
        "normal_wrn_imagenet",
    ],
    help="model architecture: " + " | ".join(model_names) + " (default: shared wide resnet)",
)
parser.add_argument(
    "--effnet_arch", metavar="ARCH", default=None, help="EfficientNet architecture type"
)
parser.add_argument(
    "--depth", type=int, metavar="N", default=28, help="Used for wrn and densenet_cifar"
)
parser.add_argument(
    "--wide",
    type=float,
    metavar="N",
    default=10,
    help="Used for growth on densenet cifar, width for wide resnet",
)

# Share params
parser.add_argument(
    "--member_template_sets",
    type=type_list("int"),
    default=[0, 1, 0, 1],
    help="default: member_id 0 uses template set 0, member_id 1 uses template set 1, member_id 2 uses template set 0, member_id 3 uses template set 1. Assume 4 members",
)

parser.add_argument(
    "--num_templates_each_set",
    type=type_list("int"),
    default=[2, 2],
    help="default: we have 2 sets (two entries), each set has 2 templates. Note that num_total_templates = sum(num_templates_each_set)",
)

parser.add_argument(
    "--net2net_epoch",
    type=int,
    default=0,
    help="0: no growing, x>=1: use net2net to grow at x-th epoch",
)

parser.add_argument("--log_model", action="store_true", default=False)

parser.add_argument("--evaluate_ensemble", action="store_true", default=False)

parser.add_argument("--member_ids", type=type_list("int"), default=[0])

parser.add_argument(
    "--growth_epochs",
    type=type_list("int"),
    default=[100],
    help="list of epochs for growing",
)
parser.add_argument(
    "--template_size",
    type=type_list("float"),
    default=[0.5, 0.5, 1.0],
    help="Size relative to the target model pre-growth",
)
parser.add_argument(
    "--ensemble_growth_type",
    type=str,
    default="diag",
    choices=["none", "diag", "column", "row"],
    help="location where the different ensemble members are",
)
parser.add_argument(
    "--ensemble_train_epochs",
    type=int,
    default=100,
    help="Number of epochs to train the second ensemble member_id before growing",
)
parser.add_argument(
    "--lr_cosine_epoch",
    type=str,
    default="reset",
    help='["reset", "continue", or a int number]',
)

parser.add_argument(
    "--warmup_member_0",
    type=int,
    default=0,
    help="only train coefs of model 0 for a few epochs, then both 0 and 1",
)

parser.add_argument(
    "--member_0_loss_weight",
    type=float,
    default=0.1,
    help="in ensemble_train_epochs epochs, we train both members 0 and 1. This is the weight of 0",
)

parser.add_argument(
    "--train_both_members",
    default=False,
    action="store_true",
    help="only train the 2nd ensemble member_id until growth (ie., in `ensemble_train_epochs` epochs). Default: we would train both 1st and 2nd until growth",
)


## for our method:
parser.add_argument(
    "--resume_member_0", type=str, default=None, help="path to the model 0 to resume"
)
parser.add_argument(
    "--resume_member_1", type=str, default=None, help="path to the model 1 to resume"
)


parser.add_argument(
    "--scale_templates_and_coefs",
    default=False,
    action="store_true",
    help="scale templates of 2nd student to match the magnitude of the first student",
)
parser.add_argument("--copy_templates_and_bn_0_to_1", default=False, action="store_true", help="")
parser.add_argument("--copy_templates_and_bn_0_to_1_noise", type=float, default=0.01, help="")
parser.add_argument(
    "--reset_scheduler",
    default=False,
    action="store_true",
    help="reset lr scheduler after training the 1st model",
)
parser.add_argument("--reset_optimizer", default=False, action="store_true")
parser.add_argument("--lr_schedule_as_member_0", default=False, action="store_true", help="")


parser.add_argument(
    "--cross_layer_sharing",
    default=False,
    action="store_true",
    help="Whether we should share parameters across similar layers",
)

# Growth
parser.add_argument("--lr_growing", type=float, default=0.0001, help="Learning rate for growing")
parser.add_argument("--small_lr", type=float, default=0.00001, help="Learning rate for growing")

parser.add_argument(
    "--lr_growing_min",
    type=float,
    default=0.00003,
    help="Min learning rate for growing",
)
parser.add_argument(
    "--reset_lr_scheduler_growing",
    default="4",
    choices=["1", "2", "3", "4", "5", "6", "none"],
    help="reset lr scheduler when training fully grown model. If 1 number is set, it is the cycle_limit in cosine lr scheduler.",
)
parser.add_argument("--reset_optimizer_growing", default=False, action="store_true")
parser.add_argument(
    "--coefs_growing",
    type=str,
    default="orthogonal",
    choices=[
        "none",
        "zero",
        "orthogonal",
        "copy",
        "random",
    ],
)
parser.add_argument("--coefs_noise_growing", type=float, default=1)
parser.add_argument(
    "--warmup_01_growing",
    type=int,
    default=0,
    help="only train coefs of model 0 and 1 for a few epochs",
)

parser.add_argument(
    "--freeze_first_last_growing",
    default=False,
    action="store_true",
    help="freeze first and last layer when growing",
)
parser.add_argument(
    "--freeze_coef_0_1_growing",
    default=False,
    action="store_true",
    help="freeze all coef of member 0 and 1 when growing",
)
parser.add_argument(
    "--data_growth",
    default=False,
    action="store_true",
    help="train half data then growth (data-growth setting)",
)

parser.add_argument(
    "--scale_templates_and_coefs_growing",
    default=False,
    action="store_true",
    help="scale templates of 2nd student to match the magnitude of the first student when growing",
)
parser.add_argument("--lr_warmup_growing", type=int, default=1, help="")
parser.add_argument(
    "--add_extra_templates_growing",
    type=int,
    default=0,
    help="add some extra templates to each set of templates, when growing",
)

# Optimization
parser.add_argument(
    "--epochs", metavar="N", type=int, default=100, help="Number of epochs to train."
)
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--eval_batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--drop_last", default=False, action="store_true", help="Drap last small batch")
parser.add_argument("--learning_rate", type=float, default=0.1, help="The Learning Rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
parser.add_argument(
    "--no_nesterov",
    default=False,
    action="store_true",
    help="Disable Nesterov momentum",
)
parser.add_argument(
    "--label_smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)"
)
parser.add_argument(
    "--optimizer",
    default="sgd",
    choices=["sgd", "rmsproptf"],
    help="Optimization algorithm (default: SGD)",
)

# default params used for swrn
parser.add_argument(
    "--scheduler_type", type=str, default="cosine", choices=["cosine", "steps", "none"]
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=None,
    help="Decrease learning rate at these epochs.",
)
parser.add_argument(
    "--gammas",
    type=float,
    nargs="+",
    default=None,
    help="LR is multiplied by gamma on schedule",
)

# parser add lr_templ0_factor
parser.add_argument("--lr_templ0_factor", type=float, default=1.0, help="")

parser.add_argument("--warmup_epochs", type=int, default=None, help="Use a linear warmup")
parser.add_argument("--base_lr", type=float, default=0.1, help="Starting learning rate")
# Step-based schedule used for EfficientNets.
parser.add_argument("--step_size", type=int, default=None, help="Step size for StepLR")
parser.add_argument("--step_gamma", type=float, default=None, help="Decay rate for StepLR")
parser.add_argument("--step_warmup", type=int, default=None, help="Number of warmup steps")

# Regularization
# default for swrn
parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay (L2 penalty).")
parser.add_argument(
    "--use_bn",
    default=False,
    action="store_true",
    help="Use the 2nd batchnorm in WRN28-10",
)
parser.add_argument(
    "--no_bn_decay",
    default=False,
    action="store_true",
    help="No weight decay on batchnorm",
)
parser.add_argument(
    "--cutout", dest="cutout", action="store_true", help="Enable cutout augmentation"
)
parser.add_argument("--ema_decay", type=float, default=None, help="Elastic model averaging decay")

# Checkpoints
parser.add_argument(
    "--print_freq",
    default=100,
    type=int,
    metavar="N",
    help="Print frequency, minibatch-wise ",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./snapshots_v2/",
    help="Folder to save checkpoints and log.",
)

## cosine_lr_warmup
parser.add_argument("--cosine_lr_warmup", default=2, type=int, help="")

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="Manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="Evaluate model on test set",
)
parser.add_argument(
    "--no_cifar_full",
    default=False,
    action="store_true",
    help="default: use all CIFAR training part to train then eval on test part. If turn this flag on, split the training part to train and val sets",
)
parser.add_argument(
    "--best_loss",
    default=False,
    action="store_true",
    help="Checkpoint best val loss instead of accuracy (default: no)",
)

# Acceleration
parser.add_argument("--ngpu", type=int, default=1, help="0 = CPU.")
parser.add_argument(
    "--workers", type=int, default=2, help="number of data loading workers (default: 2)"
)
parser.add_argument(
    "--dist",
    default=False,
    action="store_true",
    help="Use distributed training (default: no)",
)
parser.add_argument(
    "--amp",
    default=False,
    action="store_true",
    help="Use automatic mixed precision (default: no)",
)
parser.add_argument(
    "--no_dp",
    default=False,
    action="store_true",
    help="Disable using DataParallel (default: no)",
)

# Wandb log
parser.add_argument("--no_wandb", default=False, action="store_true", help="for logging with wandb")
parser.add_argument("--wandb_log_freq", type=int, default=7800, help="for logging model with wandb")

# Random seed
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument("--tag", type=str, default="", help="tag of the run, e.g., 'ours', 'target'")
parser.add_argument("--scc_id", type=str, default="", help="SCC job id")


parser.add_argument("--auto_chose_batch_size", action="store_true", help="")

args = parser.parse_args()
args.use_cuda = (args.ngpu > 0 or args.dist) and torch.cuda.is_available()

if args.auto_chose_batch_size:
    gpu_mem = utils.batch_size()
    print("gpu_mem", gpu_mem)
    if 16 < gpu_mem <= 24:
        args.batch_size = 32 * args.ngpu
    elif 24 < gpu_mem <= 32:
        args.batch_size = 64 * args.ngpu
    elif 32 < gpu_mem <= 48:
        args.batch_size = 128 * args.ngpu
    elif 48 < gpu_mem:
        args.batch_size = 256 * args.ngpu
    args.eval_batch_size = 2 * args.batch_size

if args.dist:
    import apex

# Handle mixed precision and backwards compatability.
if not hasattr(torch.cuda, "amp") or not hasattr(torch.cuda.amp, "autocast"):
    if args.amp:
        raise RuntimeError("No AMP support detected")

    # Provide dummy versions.

    def autocast(enabled=False):
        return contextlib.nullcontext()

    class GradScaler:
        def __init__(self, enabled=False):
            pass

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            pass

else:
    from torch.cuda.amp import autocast, GradScaler

if args.manualSeed is None:
    args.manualSeed = random.randint(100, 1000000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

tag = args.tag
args.save_path = os.path.join(args.save_path, args.dataset, tag, "seed_" + str(args.manualSeed))
result_png_path = os.path.join(args.save_path, "training_loss.png")

if get_world_rank() == 0:
    print(str(args))

if args.dist:
    initialize_dist(f"./init_{args.tag}")

best_acc = 0
best_los = float("inf")


def load_dataset():
    if args.dataset == "cifar10":
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == "cifar100":
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100
        num_classes = 100
    elif args.dataset not in ["imagenet", "rand_imagenet"]:
        assert False, "Unknown dataset : {}".format(args.dataset)

    if args.dataset == "cifar10" or args.dataset == "cifar100":
        # train_transform = transforms.Compose([transforms.Scale(256), transforms.RandomHorizontalFlip(), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=1, length=16))
        # test_transform = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        # Ensure only one rank downloads
        if args.dist and get_world_rank() != 0:
            torch.distributed.barrier()

        if args.evaluate or not args.no_cifar_full:
            train_data = dataset(
                args.data_path, train=True, transform=train_transform, download=True
            )
            test_data = dataset(
                args.data_path, train=False, transform=test_transform, download=True
            )

            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
            )

            if args.data_growth:
                indices = list(range(len(train_data)))
                np.random.shuffle(indices)
                split = int(0.5 * len(train_data))
                train_indices, _ = indices[:split], indices[split:]

                train_subset = torch.utils.data.Subset(train_data, train_indices)
                print("using half training set for data-growth setting")

                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_subset, num_replicas=get_world_size(), rank=get_world_rank()
                )
                train_loader = torch.utils.data.DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    sampler=train_sampler,
                    num_workers=args.workers,
                    pin_memory=True,
                )

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
        else:
            # partition training set into two instead.
            # note that test_data is defined using train=True
            train_data = dataset(
                args.data_path, train=True, transform=train_transform, download=True
            )
            test_data = dataset(args.data_path, train=True, transform=test_transform, download=True)

            indices = list(range(len(train_data)))
            np.random.shuffle(indices)
            split = int(0.9 * len(train_data))
            train_indices, test_indices = indices[:split], indices[split:]
            if args.dist:
                # Use the distributed sampler here.
                train_subset = torch.utils.data.Subset(train_data, train_indices)
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_subset, num_replicas=get_world_size(), rank=get_world_rank()
                )
                train_loader = torch.utils.data.DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    sampler=train_sampler,
                    num_workers=args.workers,
                    pin_memory=True,
                )
                test_subset = torch.utils.data.Subset(test_data, test_indices)
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_subset, num_replicas=get_world_size(), rank=get_world_rank()
                )
                test_loader = torch.utils.data.DataLoader(
                    test_subset,
                    batch_size=args.eval_batch_size,
                    sampler=test_sampler,
                    num_workers=args.workers,
                    pin_memory=True,
                )
            else:
                train_sampler = SubsetRandomSampler(train_indices)
                train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=train_sampler,
                )
                test_sampler = SubsetRandomSampler(test_indices)
                test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=args.eval_batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=test_sampler,
                )

        # Let ranks through.
        if args.dist and get_world_rank() == 0:
            torch.distributed.barrier()

    elif args.dataset == "imagenet":
        # if args.dist:
        imagenet_means = [0.485, 0.456, 0.406]
        imagenet_stdevs = [0.229, 0.224, 0.225]

        # Can just read off SSDs.
        if "efficientnet" in args.arch:
            image_size = models.efficientnet.EfficientNet.get_image_size(args.effnet_arch)
            train_transform = transforms.Compose(
                [
                    models.efficientnet.augmentations.Augmentation(
                        models.efficientnet.augmentations.get_fastautoaugment_policy()
                    ),
                    models.efficientnet.augmentations.EfficientNetRandomCrop(image_size),
                    transforms.Resize((image_size, image_size), PIL.Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                ]
            )
            test_transform = transforms.Compose(
                [
                    models.efficientnet.augmentations.EfficientNetCenterCrop(image_size),
                    transforms.Resize((image_size, image_size), PIL.Image.BICUBIC),
                ]
            )
        else:
            # Transforms adapted from imagenet_seq's, except that color jitter
            # and lighting are not applied in random orders, and that resizing
            # is done with bilinear instead of cubic interpolation.
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            test_transform = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop((224, 224))]
            )
        train_data = dset.ImageFolder(args.data_path + "/train", transform=train_transform)
        test_data = dset.ImageFolder(
            "/projectnb/ivc-ml/piotrt/data/imagenet/val", transform=test_transform
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=get_world_size(), rank=get_world_rank()
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=fast_collate,
            drop_last=args.drop_last,
        )
        train_loader = PrefetchWrapper(
            train_loader,
            imagenet_means,
            imagenet_stdevs,
            Lighting(
                0.1,
                torch.Tensor([0.2175, 0.0188, 0.0045]).cuda(),
                torch.Tensor(
                    [
                        [-0.5675, 0.7192, 0.4009],
                        [-0.5808, -0.0045, -0.8140],
                        [-0.5836, -0.6948, 0.4203],
                    ]
                ).cuda(),
            ),
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data, num_replicas=get_world_size(), rank=get_world_rank()
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=fast_collate,
        )
        test_loader = PrefetchWrapper(test_loader, imagenet_means, imagenet_stdevs, None)

        num_classes = 1000

    elif args.dataset == "rand_imagenet":
        imagenet_means = [0.485, 0.456, 0.406]
        imagenet_stdevs = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224)),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
            ]
        )
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
        train_data = RandomDataset((3, 256, 256), 1200000, pil=True, transform=train_transform)
        test_data = RandomDataset((3, 256, 256), 50000, pil=True, transform=test_transform)
        if args.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data, num_replicas=get_world_size(), rank=get_world_rank()
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data, num_replicas=get_world_size(), rank=get_world_rank()
            )
        else:
            train_sampler = RandomSampler(train_data)
            test_sampler = RandomSampler(test_data)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=fast_collate,
        )
        train_loader = PrefetchWrapper(
            train_loader,
            imagenet_means,
            imagenet_stdevs,
            Lighting(
                0.1,
                torch.Tensor([0.2175, 0.0188, 0.0045]).cuda(),
                torch.Tensor(
                    [
                        [-0.5675, 0.7192, 0.4009],
                        [-0.5808, -0.0045, -0.8140],
                        [-0.5836, -0.6948, 0.4203],
                    ]
                ).cuda(),
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=fast_collate,
        )
        test_loader = PrefetchWrapper(test_loader, imagenet_means, imagenet_stdevs, None)
        num_classes = 1000
    else:
        assert False, "Do not support dataset : {}".format(args.dataset)

    return num_classes, train_loader, test_loader


def load_model(num_classes, log):
    print_log("=> creating model '{}'".format(args.arch), log)
    if args.arch == "efficientnet_imagenet":
        net = models.efficientnet_imagenet(args.effnet_arch, args)
    else:
        net = models.__dict__[args.arch](num_classes, args)
    print_log("=> network :\n {}".format(net), log)
    if args.dist:
        net = net.to(get_cuda_device())
    else:
        net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(args.ngpu)))

    analyze_model(net)
    return net


def get_optimizer(
    net,
    state,
    lr=None,
    training_members: Optional[list] = None,
    log=None,
    optimizer=None,
    is_growing=False,
):
    def is_in_names(n, names):
        for y in names:
            if y in n:
                return True
        return False

    if args.growth_epochs[0] == -1:  ## no growing
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr,
            momentum=state["momentum"],
            weight_decay=state["decay"],
            nesterov=(not args.no_nesterov and state["momentum"] > 0.0),
        )
        return optimizer

    coef_0_names = ["coefs_member_0"]
    coef_1_names = ["coefs_member_1"]
    coef_23_names = ["coefs_member_2", "coefs_member_3"]
    teml_0_names = ["template_set_0"]
    teml_1_names = ["template_set_1"]
    bn_full_names = ["batch_norm_fully_grown"]
    bn_0_names = ["batch_norms.0"]
    bn_1_names = ["batch_norms.1"]

    coefs_0 = []
    coefs_1 = []
    coefs_23 = []
    teml_0 = []
    teml_1 = []
    bn_full = []
    bn_0 = []
    bn_1 = []
    rest = []

    _coefs_0_name, _temp0_name, _bn0_name = [], [], []
    _coefs_1_name, _temp1_name, _bn1_name = [], [], []
    _coefs_23_name = []
    i = 0

    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        elif "bank" in n:
            teml_id = str(n.split(".")[-1])
            if teml_id == "0":
                teml_0.append(p)
                _temp0_name.append(i)
                i += 1
            elif teml_id == "1":
                teml_1.append(p)
                _temp1_name.append(i)
                i += 1
            else:
                raise NotImplementedError("teml_id not supported")
        elif is_in_names(n, teml_0_names):
            teml_0.append(p)
            _temp0_name.append(i)
            i += 1

        elif is_in_names(n, teml_1_names):
            teml_1.append(p)
            _temp1_name.append(i)
            i += 1
        elif is_in_names(n, coef_0_names):
            coefs_0.append(p)
            _coefs_0_name.append(i)
            i += 1
        elif is_in_names(n, coef_1_names):
            coefs_1.append(p)
            _coefs_1_name.append(i)
            i += 1
        elif is_in_names(n, coef_23_names):
            coefs_23.append(p)
            _coefs_23_name.append(i)
            i += 1
        elif is_in_names(n, bn_full_names):
            bn_full.append(p)
            i += 1
        elif is_in_names(n, bn_0_names):
            bn_0.append(p)
            _bn0_name.append(i)
            i += 1
        elif is_in_names(n, bn_1_names):
            bn_1.append(p)
            _bn1_name.append(i)
            i += 1
        else:
            rest.append(n)
    print("ignore rest", rest)

    params = []
    if training_members is None:
        temp_0_decay, temp_1_decay = state["decay"], state["decay"]
    else:
        temp_0_decay, temp_1_decay = 0, 0
        if 0 in training_members:
            temp_0_decay = state["decay"]
        if 1 in training_members:
            temp_1_decay = state["decay"]
    lr_templ0 = lr * args.lr_templ0_factor

    params.append({"params": coefs_0, "weight_decay": 0.0, "lr": lr})
    params.append({"params": teml_0, "weight_decay": temp_0_decay, "lr": lr_templ0})
    params.append({"params": bn_0, "weight_decay": temp_0_decay, "lr": lr})

    params.append({"params": coefs_1, "weight_decay": 0.0, "lr": lr})
    params.append({"params": teml_1, "weight_decay": temp_1_decay, "lr": lr})
    params.append({"params": bn_1, "weight_decay": temp_1_decay, "lr": lr})

    params.append({"params": coefs_23, "weight_decay": 0.0, "lr": lr})

    if len(bn_full) > 0:
        params.append({"params": bn_full, "weight_decay": state["decay"], "lr": lr})

    if not args.reset_optimizer_growing and is_growing and optimizer is not None:
        print_log("add another param group in the optimizer", log)
        optimizer.add_param_group({"params": bn_full, "weight_decay": state["decay"], "lr": lr})

        ## update lr for all param_groups
        templ = True
        templ_and_bn = [1, 2, 4, 5, 7]
        for i, param_group in enumerate(optimizer.param_groups):
            if templ:
                param_group["lr"] = lr_templ0
                param_group["initial_lr"] = lr_templ0
                templ = False
            else:
                param_group["lr"] = lr
                param_group["initial_lr"] = lr

            if i in templ_and_bn:
                param_group["weight_decay"] = temp_1_decay

    else:
        print_log("make a new optimizer: get_optimizer()", log)
        optimizer = torch.optim.SGD(
            params,
            lr,
            momentum=state["momentum"],
            nesterov=(not args.no_nesterov and state["momentum"] > 0.0),
        )

    for i, param_group in enumerate(optimizer.param_groups):
        # check weight decay
        print_log(f"{i}, {param_group['weight_decay']}", log)

    return optimizer


def update_optimizer_(optimizer, member: Optional[int] = None, log=None):
    """
    Load optimizer in case of using checkpoints.
    Only used when resuming training on ImageNet.

    Optimizer's format:
    op = {
        "coef_0": op_coef0,
        "templ_mem_0": op_templ_mem0,
        "bn_0": op_bn0,
        "coef_1": op_coef1,
        "templ_mem_1": op_templ_mem1,
        "bn_1": op_bn1
    }
    """
    state_dict = optimizer.state_dict()
    if args.resume is None or args.resume == "":
        print_log("no checkpoint", log)
        return None
    else:
        checkpoint = torch.load(args.resume)
        if "optimizer_store" not in checkpoint:
            print_log("optimizer_store is not in checkpoint 'resume'", log)
            return None

    if member == 0:
        op = torch.load(args.resume_member_0)["optimizer_store"]
        cur_idx = 0
    elif member == 1:
        op = torch.load(args.resume_member_1)["optimizer_store"]
        cur_idx = 54 + 36 + 110
    else:
        op = torch.load(args.resume)["optimizer_store"]
        cur_idx = 0

    if member == 0 or member is None:
        for coef in op["coef_0"]:
            state_dict["state"][cur_idx] = coef
            cur_idx += 1
        for templ in op["templ_mem_0"]:
            state_dict["state"][cur_idx] = templ
            cur_idx += 1
        for bn in op["bn_0"]:
            state_dict["state"][cur_idx] = bn
            cur_idx += 1
        print_log("update optimizer for member 0", log)

    if member == 1 or member is None:
        for coef in op["coef_1"]:
            state_dict["state"][cur_idx] = coef
            cur_idx += 1
        for templ in op["templ_mem_1"]:
            state_dict["state"][cur_idx] = templ
            cur_idx += 1
        for bn in op["bn_1"]:
            state_dict["state"][cur_idx] = bn
            cur_idx += 1
        print_log("update optimizer for member 1", log)

    for g, m0, m1 in zip(
        range(508, 617 + 1),
        range(54 + 36, 54 + 36 + 110),
        range(54 + 36 + 110 + 54 + 36, 54 + 36 + 110 + 54 + 36 + 110),
    ):
        mem0 = state_dict["state"][m0]["momentum_buffer"]
        mem1 = state_dict["state"][m1]["momentum_buffer"]
        concat = {"momentum_buffer": torch.cat((mem0, mem1), dim=0)}
        state_dict["state"][g] = concat

    optimizer.load_state_dict(state_dict)
    return None


def log_flops(net, log, student_id=None, dataset=None):
    net_clone = copy.deepcopy(net)

    if "cifar" in dataset:
        inputs = torch.randn(1, 3, 32, 32).cuda()
    elif "imagenet" in dataset:
        inputs = torch.randn(1, 3, 224, 224).cuda()
    else:
        raise NotImplementedError

    ## FLOPS for 1 student
    student_flops = compute_flops_with_members(net_clone, inputs, member_id=student_id)
    print_log("student 1's FLOPs: {} GFLOPs".format(student_flops), log)

    ## FLOPS for 2 students: double the FLOPS
    # flops = compute_flops_with_members(net, inputs, member_id=[0, 1])
    # print_log("2 students' FLOPs: {}".format(flops), log)

    ## FLOPS of the fully grown network
    growth_update(net_clone, log=log)
    fully_grown_flops = compute_flops_with_members(net_clone, inputs, member_id=None)
    print_log("Fully grown network' FLOPs: {} GLOPs".format(fully_grown_flops), log)

    print_log(
        "FLOPs ratio student/fully_grown: {}".format(student_flops / fully_grown_flops),
        log,
    )


lr_cosine_scheduler = None
lr_cosine_epoch = -1
mylogger = None


def get_cosine_lr_scheduler(
    t_initial=None,
    optimizer=None,
    warm_up=2,
    lr_min=1.0e-06,
    cycle_limit=1,
    log=None,
    initialize=True,
):
    lr_epoch = 0
    print_log("get_cosine_lr_scheduler()", log)
    config = {
        "t_initial": t_initial,
        "lr_min": lr_min,
        "cycle_mul": 1.0,
        "cycle_decay": 0.5,
        "cycle_limit": cycle_limit,
        "warmup_t": warm_up,
        "warmup_lr_init": 1e-5,
        "warmup_prefix": False,
        "t_in_epochs": True,
        "noise_pct": 0.67,
        "noise_std": 1.0,
        "noise_seed": 42,
        "k_decay": 1.0,
        "initialize": initialize,
    }
    lr_cosine_scheduler = CosineLRScheduler(optimizer, **config)
    return lr_cosine_scheduler, lr_epoch


def freeze_coef_0_1_growing(model):
    for name, param in model.named_parameters():
        if "coefs_member_0" in name or "coefs_member_1" in name:
            param.requires_grad = False


def main():
    global best_acc, best_los, lr_cosine_scheduler, lr_cosine_epoch, mylogger

    if get_world_rank() == 0:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        log = open(
            os.path.join(
                args.save_path,
                "log_{}_seed{}.txt".format(datetime_now("%Y-%b-%d__%H-%M-%S"), args.manualSeed),
            ),
            "w",
        )
    else:
        log = None
    print_log("save path : {}".format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace("\n", " ")), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)

    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log(f"Ranks: {get_world_size()}", log)
    print_log(f"Global batch size: {args.batch_size * get_world_size()}", log)

    if get_world_rank() == 0 and not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    num_classes, train_loader, test_loader = load_dataset()

    net = load_model(num_classes, log)
    if args.growth_epochs[0] == -1:  ## normal model, no growing
        member_id = None
    else:
        member_id = [0]

    if args.member_ids:  ## override member_id
        member_id = args.member_ids
        print_log("override member_id: {}".format(member_id), log)

    if args.label_smoothing > 0.0:
        criterion = LabelSmoothingNLLLoss(args.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.optimizer == "sgd":
        if args.dist:
            decay_skip = ["coefficients"]
            if args.no_bn_decay:
                decay_skip.append("bn")
            params, _ = group_weight_decay(net, state["decay"], decay_skip)
            optimizer = apex.optimizers.FusedSGD(
                params,
                state["learning_rate"],
                momentum=state["momentum"],
                nesterov=(not args.no_nesterov and state["momentum"] > 0.0),
            )
        else:
            optimizer = get_optimizer(
                net,
                state,
                state["learning_rate"],
                training_members=args.member_ids,
                log=log,
            )

    else:
        decay_skip = ["coefficients"]
        if args.no_bn_decay:
            decay_skip.append("bn")
        params, _ = group_weight_decay(net, state["decay"], decay_skip)
        optimizer = models.efficientnet.RMSpropTF(
            params,
            state["learning_rate"],
            alpha=0.9,
            eps=1e-3,
            momentum=state["momentum"],
        )

    if args.scheduler_type == "cosine":
        print_log("setup lr_cosine_scheduler", log)
        if args.growth_epochs[0] == -1:
            n_epochs_init = args.epochs
        else:
            n_epochs_init = args.growth_epochs[0]
        lr_cosine_scheduler, _ = get_cosine_lr_scheduler(
            n_epochs_init, optimizer, log=log, warm_up=args.cosine_lr_warmup
        )

    if args.step_size:
        if args.schedule:
            raise ValueError("Cannot combine regular and step schedules")
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.step_gamma)
        if args.step_warmup:
            step_scheduler = models.efficientnet.GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                warmup_epoch=args.step_warmup,
                after_scheduler=step_scheduler,
            )
    else:
        step_scheduler = None

    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True,
        )
    scaler = GradScaler(enabled=args.amp)

    if args.ema_decay:
        ema_model = copy.deepcopy(net).to(get_cuda_device())
        ema_manager = models.efficientnet.EMA(args.ema_decay)
    else:
        ema_model, ema_manager = None, None

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == "auto":
            args.resume = os.path.join(args.save_path, "checkpoint.pth.tar")
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(
                args.resume, map_location=get_cuda_device() if args.ngpu else "cpu"
            )

            args.start_epoch = checkpoint["epoch"] if not args.start_epoch else args.start_epoch
            # Hack to load models that were wrapped in (D)DP.
            if args.no_dp:
                net = torch.nn.DataParallel(net, device_ids=[get_local_rank()])
            net.load_state_dict(checkpoint["state_dict"], strict=False)

            if args.no_dp:
                net = net.module
            if checkpoint.get("optimizer") is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    print_log("=> loaded optimizer from checkpoint", log)
                except Exception as e:
                    print(e)
                    print_log("=> didn't load optimizer from checkpoint", log)
            else:
                print_log("=> didn't load optimizer from checkpoint", log)
            if args.lr_cosine_epoch == "reset":
                lr_cosine_epoch = -1
            elif args.lr_cosine_epoch == "continue":
                lr_cosine_epoch = checkpoint.get("lr_cosine_epoch", -1)
            else:
                lr_cosine_epoch = int(args.lr_cosine_epoch)

            print_log(f"=> loaded lr_cosine_epoch={lr_cosine_epoch}", log)
            if step_scheduler:
                step_scheduler.load_state_dict(checkpoint["scheduler"])
            if ema_manager is not None:
                ema_manager.shadow = checkpoint["ema"]
            if args.amp:
                scaler.load_state_dict(checkpoint["amp"])
            best_acc = recorder.max_accuracy(False)
            print_log(
                "=> loaded checkpoint '{}' accuracy={} (epoch {})".format(
                    args.resume, best_acc, checkpoint["epoch"]
                ),
                log,
            )
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> didn't use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate_ensemble:
        checkpoint = torch.load(
            args.resume_member_0,
            map_location=get_cuda_device() if args.ngpu else "cpu",
        )
        model_0 = copy.deepcopy(net)
        model_0.load_state_dict(checkpoint["state_dict"], strict=False)

        checkpoint = torch.load(
            args.resume_member_1,
            map_location=get_cuda_device() if args.ngpu else "cpu",
        )
        model_1 = copy.deepcopy(net)
        model_1.load_state_dict(checkpoint["state_dict"], strict=False)

        validate_member(test_loader, model_0, criterion, log)
        validate_member(test_loader, model_1, criterion, log)
        print("-----\nvalidate ensemble")
        validate_ensemble(test_loader, model_0, model_1, criterion, log)
        return 0
    elif args.evaluate:
        if get_world_size() > 1:
            raise RuntimeError("Do not validate with distributed training")
        growth_update(net, evaluation=True, log=log)
        validate(test_loader, net, criterion, log)  # , member_id=[0])
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    # we assume that the first epoch is pre-growth always
    # ensemble comes up later in this version
    growth_iteration = 0
    current_growth = None

    project_name = f"grow_v2.2_{args.dataset}"
    mylogger = custom_log.MyLogging(args, net, args.scc_id, project_name=project_name)
    if not args.no_wandb:
        mylogger.log_config(args)

    warmup_01_growing = args.warmup_01_growing

    if args.resume_member_0:
        new_dict = load_member(net.state_dict(), args.resume_member_0, member_id=0)
        net.load_state_dict(new_dict)
        print_log("=> loaded member 0 from checkpoint", log)
        for k, v in net.state_dict().items():
            if "bank" in k:
                print_log(f"after loaded mem 0: norm of {k} is {v.detach().norm()}", log)

    if args.resume_member_1:
        new_dict = load_member(net.state_dict(), args.resume_member_1, member_id=1)
        net.load_state_dict(new_dict)
        print_log("=> loaded member 1 from checkpoint", log)

    if args.scale_templates_and_coefs_growing:
        print_log("scale_templates_and_coefs_growing(): scale 0 to 1", log)
        scale_templates_and_coefs(net)

    update_optimizer_(optimizer, member=None, log=log)
    if args.resume_member_0:
        update_optimizer_(optimizer, member=0, log=log)
    if args.resume_member_1:
        update_optimizer_(optimizer, member=1, log=log)

    if args.growth_epochs[0] == -1:
        pass
    else:
        print_log("before training, check on (pre-trained) model performance ", log)
        member_id_list = None if args.growth_epochs[0] == -1 else [0, 1]
        validate(test_loader, net, criterion, log, member_id=member_id_list, wandb_log=False)
        # **Test**  Prec@1 72.398 Prec@5 90.696 Error@1 27.602 Loss 1.16873
        # **Test**  Prec@1 72.230 Prec@5 90.554 Error@1 27.770 Loss 1.17870
    print("---------------------------")
    print("args.auto_chose_batch_size", args.auto_chose_batch_size)
    print("args.batch_size", args.batch_size)
    print("args.eval_batch_size", args.eval_batch_size)

    for epoch in range(args.start_epoch, args.epochs):
        mylogger.info({"lr_cosine_epoch": lr_cosine_epoch})

        lr_cosine_epoch += 1
        if step_scheduler:
            current_learning_rate = step_scheduler.get_last_lr()[0]
        else:
            current_learning_rate = adjust_learning_rate(
                optimizer, epoch, args.gammas, args.schedule, train_los
            )
        if args.growth_epochs[0] == -1:  ## normal model
            if args.net2net_epoch == 0:  # no growing
                print_log("no growing", log)
            else:
                if epoch == args.net2net_epoch:
                    print_log("..................... growing net2net", log)  ## use net2net to grow
                    if "cifar" in str(args.dataset).lower():
                        noise = False
                        net = net2net_wrn28_k_wider(net, noise=noise)
                    else:
                        noise = False
                        net = net2net_wrn_imagenet_wider(net, noise=noise)

                    if args.data_growth:  ## use full training dataset
                        args.data_growth = False
                        num_classes, train_loader, test_loader = load_dataset()

                    optimizer = get_optimizer(net, state, lr=args.lr_growing, log=log)

                    lr_cosine_scheduler, lr_cosine_epoch = get_cosine_lr_scheduler(
                        args.epochs - epoch,
                        optimizer,
                        log=log,
                        warm_up=args.cosine_lr_warmup,
                    )

                    print("reset optimizer and lr_cosine_scheduler")
                    print(net)
            member_id = None

        else:  ## template mixing
            if len(args.growth_epochs) > growth_iteration:  ## still in the growing process
                current_growth = args.growth_epochs[growth_iteration]
                if epoch == current_growth:  ## growth time
                    if args.data_growth:  ## use full training dataset
                        args.data_growth = False
                        num_classes, train_loader, test_loader = load_dataset()

                    if args.ensemble_train_epochs > 0:
                        if not args.train_both_members:
                            print_log("switching training", log)
                            member_id = [growth_iteration + 1]
                        else:
                            member_id.append(growth_iteration + 1)

                        if args.copy_templates_and_bn_0_to_1:
                            print_log("copy_templates_and_bn_0_to_1", log)
                            copy_templates_and_bn_0_to_1(net)
                        # if args.reset_optimizer:
                        optimizer = get_optimizer(
                            net,
                            state,
                            lr=args.learning_rate,
                            training_members=member_id,
                            log=log,
                        )

                        if warmup_01_growing > 0:
                            print_log(
                                "warmup_member_0: freeze_templates() of member_id 0",
                                log,
                            )
                            toggle_grad_of_template_set(
                                model=net,
                                template_set_idx=args.member_template_sets[0],
                                freeze=True,
                            )

                        if args.reset_scheduler:
                            print_log("args.reset_scheduler get_cosine_lr_scheduler()", log)
                            if args.lr_schedule_as_member_0:
                                ep = args.growth_epochs[0]
                            elif args.reset_lr_scheduler_growing == "none":
                                ep = args.epochs - epoch
                            else:
                                ep = args.ensemble_train_epochs

                            (
                                lr_cosine_scheduler,
                                lr_cosine_epoch,
                            ) = get_cosine_lr_scheduler(ep, optimizer, log=log)
                            current_learning_rate = adjust_learning_rate(
                                optimizer, epoch, args.gammas, args.schedule, train_los
                            )
                    else:
                        print_log("no ensemble training", log)

                if warmup_01_growing > 0 and epoch == warmup_01_growing + current_growth:
                    print_log("unfreeze_templates() of member_id 0", log)
                    toggle_grad_of_template_set(
                        model=net,
                        template_set_idx=args.member_template_sets[0],
                        freeze=False,
                    )

                if epoch == (
                    current_growth + args.ensemble_train_epochs
                ):  ## train a fully grown model
                    member_id = None
                    growth_update(net, log=log)

                    if args.freeze_first_last_growing:
                        print_log("freeze_first_last_growing", log)
                        for p in net.module.conv1.conv.coefficients.values():
                            p.requires_grad = False
                        for p in net.module.fc.coefficients.values():
                            p.requires_grad = False

                    if args.freeze_coef_0_1_growing:
                        print_log("freeze_coef_0_1_growing", log)
                        freeze_coef_0_1_growing(net)

                    validate(
                        test_loader,
                        net,
                        criterion,
                        log,
                        member_id=None,
                        wandb_log=False,
                    )
                    growth_iteration += 1

                    optimizer = get_optimizer(
                        net,
                        state,
                        lr=args.lr_growing,
                        training_members=None,
                        log=log,
                        optimizer=optimizer,
                        is_growing=True,
                    )

                    if (
                        args.reset_lr_scheduler_growing != "none"
                        and args.scheduler_type == "cosine"
                    ):
                        print_log(
                            "reset_lr_scheduler_growing get_cosine_lr_scheduler() ------------",
                            log,
                        )
                        print_log("change args.scheduler_type to cosine", log)
                        cycles = int(args.reset_lr_scheduler_growing)
                        lr_cosine_scheduler, lr_cosine_epoch = get_cosine_lr_scheduler(
                            (args.epochs - epoch) // cycles,
                            optimizer=optimizer,
                            warm_up=args.lr_warmup_growing,
                            cycle_limit=cycles,
                            lr_min=args.lr_growing_min,
                            log=log,
                            initialize=True,
                        )
                        current_learning_rate = adjust_learning_rate(
                            optimizer, epoch, args.gammas, args.schedule, train_los
                        )

        print_log(
            f"\nstart of epoch: {epoch + 1}, current_growth: {current_growth}, growth_iteration: {growth_iteration}, member_id: {member_id}",
            log,
        )

        mylogger.info({"lr": current_learning_rate, "epoch": epoch + 1})

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = "[Need: {:02d}:{:02d}:{:02d}]".format(need_hour, need_mins, need_secs)

        print_log(
            "\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]".format(
                time_string(), epoch + 1, args.epochs, need_time, current_learning_rate
            )
            + " [Best : Accuracy={:.2f}, Error={:.2f}]".format(
                recorder.max_accuracy(False), 100 - recorder.max_accuracy(False)
            ),
            log,
        )

        if args.dist:
            train_loader.sampler.set_epoch(epoch)
            test_loader.sampler.set_epoch(epoch)
        train_acc, train_los = train(
            train_loader,
            net,
            criterion,
            optimizer,
            scaler,
            epoch,
            log,
            step_scheduler,
            ema_manager,
            member_id,
        )
        torch.cuda.synchronize()

        val_los, val_acc = validate(
            test_loader,
            net,
            criterion,
            log,
            ema_model,
            ema_manager,
            member_id,
            epoch=epoch,
        )
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        is_best = False
        if args.best_loss:
            if val_los < best_los:
                is_best = True
                best_los = val_los
        else:
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc

        if get_world_rank() == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": net.state_dict(),
                    "recorder": recorder,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": step_scheduler.state_dict() if step_scheduler else None,
                    "lr_cosine_epoch": lr_cosine_epoch,
                    "lr_cosine_scheduler": lr_cosine_scheduler.state_dict()
                    if lr_cosine_scheduler
                    else None,
                    "ema": ema_manager.state_dict() if ema_manager is not None else None,
                    "amp": scaler.state_dict() if args.amp else None,
                },
                is_best,
                args.save_path,
                "checkpoint.pth.tar",
            )
            if (
                epoch
                in [
                    args.growth_epochs[0] - 1,
                    args.growth_epochs[0] + args.ensemble_train_epochs - 1,
                ]
                or args.log_model
            ):
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": net.state_dict(),
                        "recorder": recorder,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": step_scheduler.state_dict() if step_scheduler else None,
                        "lr_cosine_epoch": lr_cosine_epoch,
                        "lr_cosine_scheduler": lr_cosine_scheduler.state_dict()
                        if lr_cosine_scheduler
                        else None,
                        "ema": ema_manager.state_dict() if ema_manager is not None else None,
                        "amp": scaler.state_dict() if args.amp else None,
                    },
                    is_best,
                    args.save_path,
                    f"checkpoint_epoch{epoch + 1}.pth.tar",
                )
                print_log(f"save checkpoint_epoch{epoch + 1}.pth.tar", log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        if get_world_rank() == 0:
            recorder.plot_curve(result_png_path)

    mylogger.info({"Test/last_acc": val_acc})

    if get_world_rank() == 0:
        log.close()


def update_children(model, ind, evaluation):
    for layer in model.children():
        if isinstance(layer, (layers.SConv2d, layers.SLinear, layers.SBatchNorm2d)):
            layer.growth_update(ind, evaluation)
        else:
            update_children(layer, ind, evaluation)


def growth_update(model, ind=None, evaluation=False, log=None):
    print_log("--- call growth_update()", log)

    if ind is None:
        # assumes we are testing the final model
        ind = len(args.template_size) - 1

    update_children(model, ind, evaluation)

    ## update the template size
    if args.add_extra_templates_growing > 0:
        print_log(f"--- add_extra_templates_growing={args.add_extra_templates_growing}", log)
        assert args.reset_optimizer_growing
        for i in range(len(args.num_templates_each_set)):
            args.num_templates_each_set[i] += args.add_extra_templates_growing


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    log,
    step_scheduler=None,
    ema_manager=None,
    member_id=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        num_updates = lr_cosine_epoch * len(train_loader) + i
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast(enabled=args.amp):
            if member_id is None:
                output = model(input, member_id)
                loss = criterion(output, target)
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
            else:
                loss = 0.0
                prec1, prec5 = 0.0, 0.0
                for member in member_id:
                    output = model(input, member)
                    if member == 0:
                        loss += criterion(output, target) * args.member_0_loss_weight
                    else:
                        loss += criterion(output, target)
                    current_prec1, current_prec5 = accuracy(output, target, topk=(1, 5))

                    prec1 += current_prec1
                    prec5 += current_prec5

                loss /= float(len(member_id))
                prec1 /= float(len(member_id))
                prec5 /= float(len(member_id))

        if args.dist:
            reduced_loss = allreduce_tensor(loss.data)
            reduced_prec1 = allreduce_tensor(prec1)
            reduced_prec5 = allreduce_tensor(prec5)
            losses.update(reduced_loss.item(), input.size(0))
            top1.update(reduced_prec1.item(), input.size(0))
            top5.update(reduced_prec5.item(), input.size(0))
        else:
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step_scheduler:
            step_scheduler.step()
        optimizer.zero_grad()

        if lr_cosine_scheduler is not None:
            lr_cosine_scheduler.step_update(num_updates=num_updates)

        if ema_manager is not None:
            ema_manager.update(model, i)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log(
                "  Epoch: [{:03d}][{:03d}/{:03d}]   "
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})   "
                "Data {data_time.val:.3f} ({data_time.avg:.3f})   "
                "Loss {loss.val:.4f} ({loss.avg:.4f})   "
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})   "
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})   ".format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
                + time_string(),
                log,
            )
    print_log(
        "  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}".format(
            top1=top1, top5=top5, error1=100 - top1.avg
        ),
        log,
    )
    if member_id is None:
        mylogger.info(
            {
                "Train/grown/loss": losses.avg,
                "Train/grown/acc": top1.avg,
                "Train/grown/epoch": epoch + 1,
            }
        )
    else:
        mylogger.info(
            {
                "Train/model_{}/loss".format(str(member_id)): losses.avg,
                "Train/model_{}/acc".format(str(member_id)): top1.avg,
                "Train/grown/epoch": epoch + 1,
            }
        )
    if args.dist:
        torch.distributed.barrier()
    torch.cuda.synchronize()
    return top1.avg, losses.avg


def validate(
    val_loader,
    model,
    criterion,
    log,
    ema_model=None,
    ema_manager=None,
    member_id=None,
    wandb_log=True,
    epoch=None,
):
    if member_id is None:
        acc, loss = validate_member(val_loader, model, criterion, log)
        if wandb_log:
            mylogger.info(
                {
                    "Test/grown/loss": loss,
                    "Test/grown/acc": acc,
                    "Test/grown/epoch": epoch + 1,
                }
            )
        return loss, acc

    losses, acc = [], []
    for member in member_id:
        member_acc, member_loss = validate_member(
            val_loader, model, criterion, log, ema_model, ema_manager, member_id=member
        )
        if wandb_log:
            mylogger.info(
                {
                    f"Test/model_{member}/loss": member_loss,
                    f"Test/model_{member}/acc": member_acc,
                    f"Test/model_{member}/epoch": epoch + 1,
                }
            )

        losses.append(member_loss)
        acc.append(member_acc)

    return np.mean(losses), np.mean(acc)


def load_member(state_dict, member_path, member_id: int):
    """
    member_id: should be 0 or 1.
    """
    learned_dict = torch.load(member_path)["state_dict"]
    learned_keys = ["coefs_member_0", "template_set_0", "batch_norms.0"]
    for k in learned_dict.keys():
        if any([mem_0_key in k for mem_0_key in learned_keys]):
            member_k = (
                k.replace("batch_norms.0.", f"batch_norms.{member_id}.")
                .replace("template_set_0", f"template_set_{member_id}")
                .replace("coefs_member_0", f"coefs_member_{member_id}")
            )
            state_dict[member_k] = learned_dict[k]
        elif "bank" in k:
            member = str(k.split(".")[-1])
            if member == "0":
                names = k.split(".")[:-1]
                names.append(str(member_id))
                member_k = ".".join(names)
                state_dict[member_k] = learned_dict[k]

    return state_dict


def validate_ensemble(
    val_loader,
    model_1,
    model_2,
    criterion,
    log,
    ema_model=None,
    ema_manager=None,
    member_id=None,
):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if ema_model is not None:
        ema_model.module.load_state_dict(ema_manager.state_dict())
        model = ema_model

    model_1.eval()
    model_2.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.debug and i > 50:
                print_log("debug mode!!!!!", log)
                break
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with autocast(enabled=args.amp):
                output_1 = model_1(input, member_id)
                output_2 = model_2(input, member_id)
                output = (output_1 + output_2) / 2
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            if args.dist:
                reduced_loss = allreduce_tensor(loss.data)
                reduced_prec1 = allreduce_tensor(prec1)
                reduced_prec5 = allreduce_tensor(prec5)
                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))
            else:
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

    print_log(
        "  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} ".format(
            top1=top1, top5=top5, error1=100 - top1.avg, losses=losses
        ),
        log,
    )
    return top1.avg, losses.avg


def validate_member(
    val_loader, model, criterion, log, ema_model=None, ema_manager=None, member_id=None
):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if ema_model is not None:
        ema_model.module.load_state_dict(ema_manager.state_dict())
        model = ema_model

    model.eval()

    with torch.no_grad():
        for input, target in val_loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with autocast(enabled=args.amp):
                output = model(input, member_id)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            if args.dist:
                reduced_loss = allreduce_tensor(loss.data)
                reduced_prec1 = allreduce_tensor(prec1)
                reduced_prec5 = allreduce_tensor(prec5)
                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))
            else:
                losses.update(loss.data.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

    print_log(
        "  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} ".format(
            top1=top1, top5=top5, error1=100 - top1.avg, losses=losses
        ),
        log,
    )
    return top1.avg, losses.avg


def print_log(print_string, log):
    if get_world_rank() != 0:
        return  # Only print on rank 0.
    print("{}".format(print_string))
    log.write("{}\n".format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    if get_world_rank() != 0:
        return
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, "model_best.pth.tar")
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss):
    if args.scheduler_type == "steps":
        if args.warmup_epochs is not None and epoch <= args.warmup_epochs:
            incr = (args.learning_rate - args.base_lr) / args.warmup_epochs
            lr = args.base_lr + incr * epoch
        else:
            lr = args.learning_rate
            assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
            for gamma, step in zip(gammas, schedule):
                if epoch >= step:
                    lr = lr * gamma
                else:
                    break
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    elif args.scheduler_type == "cosine":
        lr_cosine_scheduler.step(lr_cosine_epoch)
        if args.net2net_epoch > 0:
            lr = optimizer.param_groups[0]["lr"]
            print("lr={}".format(lr))
        else:
            try:
                lr = (optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
                print("lr={}".format(lr))
                lr = lr[1]
            except:
                lr = optimizer.param_groups[0]["lr"]
                print("lr={}".format(lr))

    elif args.scheduler_type == "none":
        lr = args.learning_rate
    else:
        raise NotImplementedError("Only support steps and cosine scheduler")

    return lr


def toggle_grad_of_template_set(model, template_set_idx: int, freeze=True):
    requires_grad = not freeze
    for name, param in model.named_parameters():
        if f"template_set_{template_set_idx}" in name:
            param.requires_grad = requires_grad


def group_weight_decay(
    net,
    weight_decay,
    skip_list=(),
    template_sets_decay: Dict = None,
    bn_full_decay: float = None,
    log=None,
):
    """
    template_sets_decay: {template_set_i: decay_value}
        example for template_sets_decay: {"template_set_0": 0, "template_set_1": 1e-4}
    """
    decay, no_decay = [], []
    temp_sets = defaultdict(list)
    bn_full = []
    # decay_names, no_decay_names = [], []
    ind = 0
    tem1_id, tem0_id = [], []
    coef1_id, coef0_id = [], []
    bn1_id, bn0_id = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        ## pick up template set parameters first
        elif (template_sets_decay is not None) and (
            "template_set" in name or "batch_norms" in name or "bank" in name
        ):
            if "template_set" in name or "bank" in name:
                temp_set_i = name.split(".")[-1]
                temp_sets[temp_set_i].append((name, param))
                if temp_set_i == "0":
                    tem0_id.append(ind)
                else:
                    tem1_id.append(ind)
                ind += 1

                print_log(f"-added template_set{temp_set_i} {name}", log)
            elif "batch_norms" in name:
                i = name.split(".")[-2]
                temp_sets[f"template_set_{i}"].append((name, param))

                print_log(f"-added template_set_{i} {name}", log)
                if i == "0":
                    bn0_id.append(ind)
                else:
                    bn1_id.append(ind)
                ind += 1
            else:
                raise NotImplementedError
        elif bn_full_decay is not None and "batch_norm_fully_grown" in name:
            bn_full.append((name, param))
            ind += 1
        elif sum([pattern in name for pattern in skip_list]) > 0:
            no_decay.append((name, param))
            print_log(f"-added to no_decay {name}", log)
            if "coefs_member_0" in name:
                coef0_id.append(ind)
                ind += 1
            elif "coefs_member_1" in name:
                coef1_id.append(ind)
                ind += 1
            else:
                ind += 1
        else:
            decay.append((name, param))
            print_log(f"-added to decay {name}", log)
            ind += 1

    tmp = {}
    temp_set_i_list = None
    if template_sets_decay is not None:
        temp_set_i_list = sorted(template_sets_decay.keys())
        for temp_set_i in temp_set_i_list:
            decay_value = template_sets_decay[temp_set_i]
            tmp[temp_set_i] = {
                "params": [x[1] for x in temp_sets[temp_set_i]],
                "weight_decay": decay_value,
            }

    res = []
    if tmp and args.growth_epochs[0] != -1:
        res.extend(tmp.values())
        print_log(f"group_weight_decay(): template_sets, len(res)={len(res)}", log)

    res.append({"params": [x[1] for x in no_decay], "weight_decay": 0.0})

    res.append({"params": [x[1] for x in decay], "weight_decay": weight_decay})

    print_log(f"group_weight_decay(): no_decay, decay, len(res)={len(res)}", log)

    if bn_full_decay is not None and args.growth_epochs[0] != -1:
        res.append({"params": [x[1] for x in bn_full], "weight_decay": bn_full_decay})
        print_log(f"group_weight_decay(): bn_full, len(res)={len(res)}", log)

    return res, (temp_set_i_list, bn_full)


def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1:
        return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
