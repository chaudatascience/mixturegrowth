import torch
import numpy as np
from collections import Counter
from utils import get_module_by_name, set_module_by_name_
import copy


def check_norm_of_all_layers(model):
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            print(name, module.weight.norm().item())


## https://github.com/erogol/Net2Net/blob/fffc2b66df8a11577518f7f01287abe264ce30de/net2net.py#L118


@torch.no_grad()
def expand_wider(
    model,
    m1_name,
    m2_name,
    bnorm_name="",
    factor=2,
    out_size=None,
    noise=False,
    random_init=False,
    weight_norm=True,
    update_m1=True,
    update_m2=True,
    update_bn=True,
):
    m1 = copy.deepcopy(get_module_by_name(model, m1_name))
    m2 = copy.deepcopy(get_module_by_name(model, m2_name))

    if bnorm_name != "":
        bnorm = copy.deepcopy(get_module_by_name(model, bnorm_name))
    else:
        bnorm = None

    ## compare size of original vs model

    new_width = m1.weight.shape[0] * factor

    # if m2_name == "group_2.0.shortcut":
    # breakpoint()
    new_m1, new_m2, new_bnorm = wider(
        m1, m2, new_width, bnorm, out_size, noise, random_init, weight_norm
    )
    if update_m1:
        set_module_by_name_(model, m1_name, new_m1)
    if update_m2:
        set_module_by_name_(model, m2_name, new_m2)
    if update_bn:
        if bnorm_name != "":
            model = set_module_by_name_(model, bnorm_name, new_bnorm)
    return None


def wider(
    m1, m2, new_width, bnorm=None, out_size=None, noise=True, random_init=True, weight_norm=True
):
    """
    Convert m1 layer to its wider version by adapthing next weight layer and
    possible batch norm layer in btw.
    Args:
        m1 - module to be wider
        m2 - follwing module to be adapted to m1
        new_width - new width for m1.
        bn (optional) - batch norm layer, if there is btw m1 and m2
        out_size (list, optional) - necessary for m1 == conv3d and m2 == linear. It
            is 3rd dim size of the output feature map of m1. Used to compute
            the matching Linear layer size
        noise (bool, True) - add a slight noise to break symmetry btw weights.
        random_init (optional, True) - if True, new weights are initialized
            randomly.
        weight_norm (optional, True) - If True, weights are normalized before
            transfering.
    """
    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data if m1.bias is not None else None
    if isinstance(bnorm, torch.nn.Identity):
        bnorm = None

    if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1) // factor**2, factor, factor)
            elif w1.dim() == 5:
                assert out_size is not None, "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(
                    w2.size(0), w2.size(1) // factor, out_size[0], out_size[1], out_size[2]
                )
        else:
            assert w1.size(0) == w2.size(
                1
            ), f"{w1.shape}, {w2.shape}: Module weights are not compatible"
        assert new_width >= w1.size(0), "New size should be larger"

        old_width = w1.size(0)
        nw1 = m1.weight.data.clone()
        nw2 = w2.clone()

        if nw1.dim() == 4:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))
        elif nw1.dim() == 5:
            nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3), nw1.size(4))
            nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3), nw2.size(4))
        else:
            nw1.resize_(new_width, nw1.size(1))
            nw2.resize_(nw2.size(0), new_width)

        if b1 is not None:
            nb1 = m1.bias.data.clone()
            nb1.resize_(new_width)

        if bnorm is not None:
            nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
            nrunning_var = bnorm.running_var.clone().resize_(new_width)
            if bnorm.affine:
                nweight = bnorm.weight.data.clone().resize_(new_width)
                nbias = bnorm.bias.data.clone().resize_(new_width)

        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)

        nw1.narrow(0, 0, old_width).copy_(w1)
        nw2.narrow(0, 0, old_width).copy_(w2)
        if b1 is not None:
            nb1.narrow(0, 0, old_width).copy_(b1)

        if bnorm is not None:
            nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
            nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
            if bnorm.affine:
                nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

        # TEST:normalize weights
        if weight_norm:
            for i in range(old_width):
                norm = w1.select(0, i).norm()
                w1.select(0, i).div_(norm)

        # select weights randomly
        tracking = dict()
        for i in range(old_width, new_width):
            idx = np.random.randint(0, old_width)
            try:
                tracking[idx].append(i)
            except:
                tracking[idx] = [idx]
                tracking[idx].append(i)

            # TEST:random init for new units
            if random_init:
                n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                if m2.weight.dim() == 4:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                elif m2.weight.dim() == 5:
                    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                elif m2.weight.dim() == 2:
                    n2 = m2.out_features * m2.in_features
                nw1.select(0, i).normal_(0, np.sqrt(2.0 / n))
                nw2.select(0, i).normal_(0, np.sqrt(2.0 / n2))
            else:
                nw1.select(0, i).copy_(w1.select(0, idx).clone())
                nw2.select(0, i).copy_(w2.select(0, idx).clone())
            if b1 is not None:
                nb1[i] = b1[idx]

            if bnorm is not None:  ## is this a bug?
                nrunning_mean[i] = bnorm.running_mean[idx]
                nrunning_var[i] = bnorm.running_var[idx]
                if bnorm.affine:
                    nweight[i] = bnorm.weight.data[idx]
                    nbias[i] = bnorm.bias.data[idx]
                bnorm.num_features = new_width

        if not random_init:
            for idx, d in tracking.items():
                for item in d:
                    nw2[item].div_(len(d))

        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m1.out_channels = new_width
        m2.in_channels = new_width

        if noise:
            # noise = np.random.normal(scale=5e-2 * nw1.std(), size=list(nw1.size()))
            noise = torch.normal(mean=0, std=5e-2 * nw1.std(), size=list(nw1.size()), device="cuda")
            nw1 += noise  # torch.FloatTensor(noise).type_as(nw1)

        m1.weight.data = nw1

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            if w1.dim() == 4:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width * factor**2)
                m2.in_features = new_width * factor**2
            elif w2.dim() == 5:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width * factor)
                m2.in_features = new_width * factor
        else:
            m2.weight.data = nw2
        if b1 is not None:
            m1.bias.data = nb1

        if bnorm is not None:
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias
        return m1, m2, bnorm


# TODO: Consider adding noise to new layer as wider operator.
def deeper(m, nonlin, bnorm_flag=False, weight_norm=True, noise=True):
    """
    Deeper operator adding a new layer on topf of the given layer.
    Args:
        m (module) - module to add a new layer onto.
        nonlin (module) - non-linearity to be used for the new layer.
        bnorm_flag (bool, False) - whether add a batch normalization btw.
        weight_norm (bool, True) - if True, normalize weights of m before
            adding a new layer.
        noise (bool, True) - if True, add noise to the new layer weights.
    """

    if "Linear" in m.__class__.__name__:
        m2 = torch.nn.Linear(m.out_features, m.out_features)
        m2.weight.data.copy_(torch.eye(m.out_features))
        m2.bias.data.zero_()

        if bnorm_flag:
            bnorm = torch.nn.BatchNorm1d(m2.weight.size(1))
            bnorm.weight.data.fill_(1)
            bnorm.bias.data.fill_(0)
            bnorm.running_mean.fill_(0)
            bnorm.running_var.fill_(1)

    elif "Conv" in m.__class__.__name__:
        assert m.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"

        if m.weight.dim() == 4:
            pad_h = int((m.kernel_size[0] - 1) / 2)
            # pad_w = pad_h
            m2 = torch.nn.Conv2d(
                m.out_channels, m.out_channels, kernel_size=m.kernel_size, padding=pad_h
            )
            m2.weight.data.zero_()
            ## https://github.com/erogol/Net2Net/issues/3
            # c = m.kernel_size[0] // 2 + 1
            c = m.kernel_size[0] // 2

        elif m.weight.dim() == 5:
            pad_hw = int((m.kernel_size[1] - 1) / 2)  # pad height and width
            pad_d = int((m.kernel_size[0] - 1) / 2)  # pad depth
            m2 = torch.nn.Conv3d(
                m.out_channels,
                m.out_channels,
                kernel_size=m.kernel_size,
                padding=(pad_d, pad_hw, pad_hw),
            )
            c_wh = m.kernel_size[1] // 2 + 1
            c_d = m.kernel_size[0] // 2 + 1

        restore = False
        if m2.weight.dim() == 2:
            restore = True
            m2.weight.data = m2.weight.data.view(
                m2.weight.size(0), m2.in_channels, m2.kernel_size[0], m2.kernel_size[0]
            )

        if weight_norm:
            for i in range(m.out_channels):
                weight = m.weight.data
                norm = weight.select(0, i).norm()
                weight.div_(norm)
                m.weight.data = weight

        for i in range(0, m.out_channels):
            if m.weight.dim() == 4:
                m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c, 1).narrow(
                    3, c, 1
                ).fill_(1)
            elif m.weight.dim() == 5:
                m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c_d, 1).narrow(
                    3, c_wh, 1
                ).narrow(4, c_wh, 1).fill_(1)

        if noise:
            noise = np.random.normal(scale=5e-2 * m2.weight.data.std(), size=list(m2.weight.size()))
            m2.weight.data += torch.FloatTensor(noise).type_as(m2.weight.data)

        if restore:
            m2.weight.data = m2.weight.data.view(
                m2.weight.size(0), m2.in_channels, m2.kernel_size[0], m2.kernel_size[0]
            )

        m2.bias.data.zero_()

        if bnorm_flag:
            if m.weight.dim() == 4:
                bnorm = torch.nn.BatchNorm2d(m2.out_channels)
            elif m.weight.dim() == 5:
                bnorm = torch.nn.BatchNorm3d(m2.out_channels)
            bnorm.weight.data.fill_(1)
            bnorm.bias.data.fill_(0)
            bnorm.running_mean.fill_(0)
            bnorm.running_var.fill_(1)

    else:
        raise RuntimeError("{} Module not supported".format(m.__class__.__name__))

    s = torch.nn.Sequential()
    s.add_module("conv", m)
    if bnorm_flag:
        s.add_module("bnorm", bnorm)
    if nonlin is not None:
        s.add_module("nonlin", nonlin())
    s.add_module("conv_new", m2)
    return s


@torch.no_grad()
def net2net_wrn28_k_wider(model, out_size=None, noise=False, random_init=False, weight_norm=True):
    original_model = copy.deepcopy(model)
    print("check1")
    check_norm_of_all_layers(model)

    layer_names = []
    layer_names_with_shortcut = []
    for name, _ in model.named_parameters():
        if "shortcut" in name:
            layer = name.replace(".weight", "").replace(".bias", "")
            if layer not in layer_names_with_shortcut:
                layer_names_with_shortcut.append(layer)
        elif any(x in name for x in ["conv", "bn", "fc", "last_act.0"]):
            layer = name.replace(".weight", "").replace(".bias", "")
            if layer not in layer_names:
                layer_names.append(layer)
                layer_names_with_shortcut.append(layer)

    skip_forward = (
        [
            ("module.group_1.3.conv2", "module.group_2.0.bn1", "module.group_2.0.shortcut"),
            ("module.group_2.3.conv2", "module.group_3.0.bn1", "module.group_3.0.shortcut"),
        ]
        if isinstance(model, torch.nn.DataParallel)
        else [
            ("group_1.3.conv2", "group_2.0.bn1", "group_2.0.shortcut"),
            ("group_2.3.conv2", "group_3.0.bn1", "group_3.0.shortcut"),
        ]
    )
    for conv, bn, shortcut in skip_forward:
        expand_wider(
            model,
            m1_name=conv,
            m2_name=shortcut,
            bnorm_name=bn,
            out_size=out_size,
            noise=noise,
            random_init=random_init,
            weight_norm=weight_norm,
            update_m1=False,
            update_bn=False,
        )

    for i in range(1, len(layer_names_with_shortcut) - 2):
        if (
            "shortcut" in layer_names_with_shortcut[i]
            and "bn" in layer_names_with_shortcut[i + 1]
            and "conv" in layer_names_with_shortcut[i + 2]
        ):
            m1_name = layer_names_with_shortcut[i]
            bn_name = layer_names_with_shortcut[i + 1]
            m2_name = layer_names_with_shortcut[i + 2]
            expand_wider(
                model,
                m1_name=m1_name,
                m2_name=m2_name,
                bnorm_name=bn_name,
                out_size=out_size,
                noise=noise,
                random_init=random_init,
                weight_norm=weight_norm,
                update_m2=False,
                update_bn=False,
            )

    for i in range(1, len(layer_names) - 1):
        if "conv" in layer_names[i]:
            m1_name = layer_names[i]
            if "bn" in layer_names[i + 1] or "last_act.0" in layer_names[i + 1]:
                bn_name = layer_names[i + 1]
                if "conv" in layer_names[i + 2] or "fc" in layer_names[i + 2]:
                    m2_name = layer_names[i + 2]
                else:
                    continue
            elif "conv" in layer_names[i + 1] or "fc" in layer_names[i + 1]:
                m2_name = layer_names[i + 1]
                bn_name = ""
            else:
                continue

            expand_wider(
                model,
                m1_name=m1_name,
                m2_name=m2_name,
                bnorm_name=bn_name,
                out_size=out_size,
                noise=noise,
                random_init=random_init,
                weight_norm=weight_norm,
            )

    print("after growing")
    check_norm_of_all_layers(model)

    return model


def net2net_wrn_imagenet_wider(
    model, out_size=None, noise=False, random_init=False, weight_norm=True
):
    layers = []
    layer_names_with_shortcut = []
    for name, _ in model.named_parameters():
        if "skip_connection" in name:
            layer = name.replace(".weight", "").replace(".bias", "")
            if layer not in layer_names_with_shortcut:
                layer_names_with_shortcut.append(layer)
        elif any(x in name for x in ["conv", "bn", "fc"]) and "skip_connection" not in name:
            layer = name.replace(".weight", "").replace(".bias", "")
            if layer not in layers:
                layers.append(layer)
                layer_names_with_shortcut.append(layer)

    get_name = lambda x: x.split(".")[-1]
    layer_names = layer_names_with_shortcut
    ## skip_connection, conv vs skip.conv
    for i in range(1, len(layer_names) - 2):
        name = get_name(layer_names[i])
        if (
            "conv" in get_name(layer_names[i])
            and "bn" in get_name(layer_names[i + 1])
            and "conv" in get_name(layer_names[i + 2])
            and "skip_connection" in layer_names[i + 2]
        ):
            m1_name = layer_names[i]
            bn_name = layer_names[i + 1]
            m2_name = layer_names[i + 2]
        elif (
            "conv" in get_name(layer_names[i])
            and "conv" in get_name(layer_names[i + 1])
            and "skip_connection" in layer_names[i + 1]
        ):
            m1_name = layer_names[i]
            bn_name = ""
            m2_name = layer_names[i + 1]
        else:
            continue

        expand_wider(
            model,
            m1_name=m1_name,
            m2_name=m2_name,
            bnorm_name=bn_name,
            noise=noise,
            random_init=random_init,
            weight_norm=weight_norm,
            update_m1=False,
            out_size=out_size,
            update_bn=("skip_connection" in bn_name),
        )

    for i in range(1, len(layer_names) - 2):
        name = get_name(layer_names[i])
        if (
            "conv" in get_name(layer_names[i])
            and "skip_connection" in layer_names[i]
            and "bn" in get_name(layer_names[i + 1])
            and "conv" in get_name(layer_names[i + 2])
        ):
            m1_name = layer_names[i]
            bn_name = layer_names[i + 1]
            m2_name = layer_names[i + 2]
        elif (
            "conv" in get_name(layer_names[i])
            and "skip_connection" in layer_names[i]
            and "conv" in get_name(layer_names[i + 1])
        ):
            m1_name = layer_names[i]
            bn_name = ""
            m2_name = layer_names[i + 1]
        else:
            continue

        block = int(m2_name.split("block_modules.")[-1].split(".")[1])
        m2_name = m2_name.replace(f".{block}.in_conv", f".{block+1}.in_conv")
        expand_wider(
            model,
            m1_name=m1_name,
            m2_name=m2_name,
            bnorm_name=bn_name,
            noise=noise,
            random_init=random_init,
            weight_norm=weight_norm,
            update_m2=False,
            out_size=out_size,
            update_bn=("skip_connection" in bn_name),
        )

    for i in range(0, len(layers) - 1):
        if "conv" in get_name(layers[i]):
            m1_name = layers[i]
            if "bn" in get_name(layers[i + 1]):
                bn_name = layers[i + 1]
                if "conv" in get_name(layers[i + 2]) or "fc" in get_name(layers[i + 2]):
                    m2_name = layers[i + 2]
                else:
                    continue
            elif "conv" in get_name(layers[i + 1]) or "fc" in get_name(layers[i + 1]):
                m2_name = layers[i + 1]
                bn_name = ""
            else:
                continue
            expand_wider(
                model,
                m1_name=m1_name,
                m2_name=m2_name,
                bnorm_name=bn_name,
                noise=noise,
                random_init=random_init,
                weight_norm=weight_norm,
                out_size=out_size,
            )
    return model


def test_():
    from torch import nn

    m1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    m2 = nn.Conv2d(4, 5, kernel_size=3, padding=1)
    bn = None
    m1.weight.data.fill_(3)
    m2.weight.data.fill_(2)
    ## show m1, m2 before and after
    print(m1.weight)
    print(m2.weight)
    wider(m1, m2, 8, bn, noise=False, random_init=True, weight_norm=True)
    print("after")
    print(m1.weight.shape)
    print(m1.weight)
    print("---")
    print(m2.weight.shape)
    print(m2.weight)


if __name__ == "__main__":
    test_()
