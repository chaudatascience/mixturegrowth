import os
import pathlib
import sys
import time
import os.path
import warnings
import random
from typing import Dict

import torch
import torchvision
import numpy as np
import matplotlib

from typing import Union
from functools import reduce

matplotlib.use("agg")
import matplotlib.pyplot as plt


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )
        return img.add(rgb.view(3, 1, 1).expand_as(img))


# Adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/smoothing.py
class LabelSmoothingNLLLoss(torch.nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = (-logprobs.gather(dim=-1, index=target.unsqueeze(1))).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class RandomDataset(torch.utils.data.Dataset):
    """Dataset that just returns a random tensor for debugging."""

    def __init__(self, sample_shape, dataset_size, label=True, pil=False, transform=None):
        super().__init__()
        self.sample_shape = sample_shape
        self.dataset_size = dataset_size
        self.label = label
        self.transform = transform
        if pil:
            d = torch.rand(sample_shape)
            self.d = torchvision.transforms.functional.to_pil_image(d)
        else:
            self.d = torch.rand(sample_shape)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        d = self.d
        if self.transform is not None:
            d = self.transform(d)
        if self.label:
            return d, 0
        else:
            return d


# Adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L250
class PrefetchWrapper:
    """Fetch ahead and do some asynchronous processing."""

    def __init__(self, data_loader, mean, stdev, lighting):
        self.data_loader = data_loader
        self.mean = mean
        self.stdev = stdev
        self.lighting = lighting
        self.stream = torch.cuda.Stream()
        self.sampler = data_loader.sampler  # To simplify set_epoch.

    def prefetch_loader(data_loader, mean, stdev, lighting, stream):
        if lighting is not None:
            mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
            stdev = torch.tensor(stdev).cuda().view(1, 3, 1, 1)
        else:
            mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
            stdev = torch.tensor([x * 255 for x in stdev]).cuda().view(1, 3, 1, 1)

        first = True
        for next_input, next_target in data_loader:
            with torch.cuda.stream(stream):
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.cuda(non_blocking=True).float()
                if lighting is not None:
                    # Scale and apply lighting first.
                    next_input = next_input.div_(255.0)
                    next_input = lighting(next_input).sub_(mean).div_(stdev)
                else:
                    next_input = next_input.sub_(mean).div_(stdev)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target
        yield input, target

    def __iter__(self):
        return PrefetchWrapper.prefetch_loader(
            self.data_loader, self.mean, self.stdev, self.lighting, self.stream
        )

    def __len__(self):
        return len(self.data_loader)


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        # Suppress warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros(
            (self.total_epoch, 2), dtype=np.float32
        )  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def refresh(self, epochs):
        if epochs == self.total_epoch:
            return
        self.epoch_losses = np.vstack(
            (
                self.epoch_losses,
                np.zeros((epochs - self.total_epoch, 2), dtype=np.float32) - 1,
            )
        )
        self.epoch_accuracy = np.vstack(
            (
                self.epoch_accuracy,
                np.zeros((epochs - self.total_epoch, 2), dtype=np.float32),
            )
        )
        self.total_epoch = epochs

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        assert (
            idx >= 0 and idx < self.total_epoch
        ), "total_epoch : {} , but update with the {} index".format(self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1
        return self.max_accuracy(False) == val_acc

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0:
            return 0
        if istrain:
            return self.epoch_accuracy[: self.current_epoch, 0].max()
        else:
            return self.epoch_accuracy[: self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = "the accuracy/loss curve of train/val"
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel("the training epoch", fontsize=16)
        plt.ylabel("accuracy", fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color="g", linestyle="-", label="train-accuracy", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color="y", linestyle="-", label="valid-accuracy", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis * 50, color="g", linestyle=":", label="train-loss-x50", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis * 50, color="y", linestyle=":", label="valid-loss-x50", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print("---- save figure {} into {}".format(title, save_path))
        plt.close(fig)


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


def datetime_now(time_format: str = None) -> str:
    from datetime import datetime

    # time_format = default(time_format, "%Y-%b-%d %H:%M:%S.%f")
    time_format = default(time_format, "%Y-%b-%d %H:%M:%S")
    return datetime.now().strftime(time_format)


def time_string():
    # ISOTIMEFORMAT = '%Y-%m-%d %X'
    # string = '[{}]'.format(time.strftime(
    #     ISOTIMEFORMAT, time.gmtime(time.time())))
    # return string
    return datetime_now()


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = "%Y-%m-%d"
    string = "{}".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string + "-{}".format(random.randint(1, 10000))


# Utilities for distributed training.


def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank():
    """Get local rank from environment."""
    if "MV2_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


def get_local_size():
    """Get local size from environment."""
    if "MV2_COMM_WORLD_LOCAL_SIZE" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_LOCAL_SIZE"])
    elif "OMPI_COMM_WORLD_LOCAL_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
    elif "SLURM_NTASKS_PER_NODE" in os.environ:
        return int(os.environ["SLURM_NTASKS_PER_NODE"])
    else:
        return 1


def get_world_rank():
    """Get rank in world from environment."""
    if "MV2_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_RANK"])
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


def get_world_size():
    """Get world size from environment."""
    if "MV2_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["MV2_COMM_WORLD_SIZE"])
    elif "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


def initialize_dist(init_file):
    """Initialize PyTorch distributed backend."""
    torch.cuda.init()
    torch.cuda.set_device(get_local_rank())
    init_file = os.path.abspath(init_file)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=get_world_rank(),
        world_size=get_world_size(),
    )
    torch.distributed.barrier()
    # Ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file):
        os.unlink(init_file)


def get_cuda_device():
    """Get this rank's CUDA device."""
    return torch.device(f"cuda:{get_local_rank()}")


def allreduce_tensor(t):
    """Allreduce and average tensor t."""
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt


def datetime_now(time_format: str = None) -> str:
    from datetime import datetime

    # time_format = default(time_format, "%Y-%b-%d %H:%M:%S.%f")
    time_format = default(time_format, "%Y-%b-%d %H:%M:%S")
    return datetime.now().strftime(time_format)


def analyze_model(model, print_out=True):
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(list(model.state_dict().keys()))

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, param in model.named_parameters():
        if print_out:
            print(name, param.shape, param.requires_grad)
    if print_out:
        print(f"\nTotal parameters: {total_num:,};\tTrainable: {trainable_num:,}")
    return total_num, trainable_num


def mkdir(path, mode=0o700):
    pathlib.Path(path).mkdir(mode=mode, parents=True, exist_ok=True)


def get_machine_name():
    import socket

    machine_name = socket.gethostname()
    return machine_name




def get_device(cuda_device=None, verbose=True):
    cuda = default(cuda_device, "cuda")
    device = torch.device(f"{cuda_device}" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("device: ", device)
    return device


def dict2obj(my_dict: Dict):
    class Obj:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return Obj(**my_dict)


def type_list(type_: str):
    if type_ == "int":
        return lambda s: [int(x) for x in s.split(",")]
    elif type_ == "float":
        return lambda s: [float(x) for x in s.split(",")]
    else:
        raise NotImplementedError()


def batch_size(cuda="cuda:0"):
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    return total_gb


def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module], access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    """
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def set_module_by_name_(
    module: Union[torch.Tensor, torch.nn.Module], access_string: str, new_value
):
    ## replace access_string by new_value
    names = access_string.split(sep=".")
    if len(names) == 1:
        setattr(module, access_string, new_value)
    else:
        my_module = get_module_by_name(module, ".".join(names[:-1]))
        setattr(my_module, names[-1], new_value)
