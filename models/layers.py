from __future__ import annotations

import contextlib
import operator
from collections import defaultdict
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, Union, TypeVar, Iterator, OrderedDict, Optional
from torch._jit_internal import _copy_to_script_wrapper

import utils

# Handle mixed precision compatibility.
if not hasattr(torch.cuda, "amp") or not hasattr(torch.cuda.amp, "autocast"):

    def autocast(enabled=False):
        return contextlib.nullcontext()

else:
    from torch.cuda.amp import autocast


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class SBatchNorm2d(nn.Module):
    def __init__(self, args, target_planes):
        super(SBatchNorm2d, self).__init__()
        self.target_planes = target_planes
        self.args = args
        # smallest_planes = int(target_planes * args.template_size[0])
        num_templates_sets = len(self.args.num_templates_each_set)
        templateset2member = {}
        for i, temp_set in enumerate(reversed(args.member_template_sets)):
            member = len(args.member_template_sets) - i - 1
            templateset2member[temp_set] = member

        self.batch_norms = nn.ModuleList()
        for i in range(num_templates_sets):
            ind = templateset2member[i]
            self.batch_norms.append(
                nn.BatchNorm2d(int(target_planes * args.template_size[ind]))
            )

        ## todo: fix later
        self.batch_norm_fully_grown = None

        self.growth_update(0)

    def growth_update(self, ind, evaluation=False):
        in_planes = int(self.target_planes * self.args.template_size[ind])
        self.in_planes = in_planes
        self.num_splits = int(np.ceil(self.target_planes / float(in_planes)))

        if self.num_splits == 1:  ## fully grown
            self.bn_growth()

    def copy_bn_0_to_1(self):
        self.batch_norms[1].weight.data = (
            self.batch_norms[0].weight.data.detach().clone()
        )
        self.batch_norms[1].bias.data = self.batch_norms[0].bias.data.detach().clone()
        self.batch_norms[1].running_mean.data = (
            self.batch_norms[0].running_mean.data.detach().clone()
        )
        self.batch_norms[1].running_var.data = (
            self.batch_norms[0].running_var.data.detach().clone()
        )

    def bn_growth(self):
        """
        create a new fully grown batch norm
        """
        weight, bias, running_mean, running_var = [], [], [], []
        for member_bn in self.batch_norms:
            weight.append(member_bn.weight.data)
            bias.append(member_bn.bias.data)
            running_mean.append(member_bn.running_mean.data)
            running_var.append(member_bn.running_var.data)
        device = self.batch_norms[0].weight.device
        ## padding 1 or 0 for the middle part
        num_channels = sum([len(w) for w in weight])

        if num_channels != self.target_planes:
            weight.insert(
                1, torch.ones(self.target_planes - num_channels, device=device)
            )
            bias.insert(
                1, torch.zeros(self.target_planes - num_channels, device=device)
            )
            running_mean.insert(
                1, torch.zeros(self.target_planes - num_channels, device=device)
            )
            running_var.insert(
                1, torch.ones(self.target_planes - num_channels, device=device)
            )

        self.batch_norm_fully_grown = nn.BatchNorm2d(self.target_planes, device=device)
        self.batch_norm_fully_grown.weight.data = torch.concat(weight, dim=0)
        self.batch_norm_fully_grown.bias.data = torch.concat(bias, dim=0)
        self.batch_norm_fully_grown.running_mean.data = torch.concat(
            running_mean, dim=0
        )
        self.batch_norm_fully_grown.running_var.data = torch.concat(running_var, dim=0)

        print(
            "--- fully grown norm",
            self.batch_norm_fully_grown.weight.data.detach().norm(),
            self.batch_norm_fully_grown.bias.data.detach().norm(),
        )
        print(
            "bn0 norm",
            self.batch_norms[0].weight.data.detach().norm(),
            self.batch_norms[0].bias.data.detach().norm(),
        )
        print(
            "bn1 norm",
            self.batch_norms[1].weight.data.detach().norm(),
            self.batch_norms[1].bias.data.detach().norm(),
        )
        self.batch_norms = None

    def forward(self, x, member_id):
        if member_id is None:  ## fully grown
            # if self.args.debug:
            #     print("--------batch_norm_fully_grown------")
            return self.batch_norm_fully_grown(x)
        else:
            templateset_idx = self.args.member_template_sets[member_id]
            # if self.args.debug:
            #     print(f"--------batch_norms[{templateset_idx}]------")
            return self.batch_norms[templateset_idx](x)


class SModule(nn.Module):
    def __init__(
        self,
        args,
        num_shared_layers: int,
        in_features: int,
        out_features: int,
        groups: int = 1,
        set_input=False,
        set_output=False,
        layer_id: Optional[int] = None,
        bank=None,
        bank_cfg: Dict = None,
    ):
        """
        args: the arguments object
        num_shared_layers: the number of shared layers in the module, either 1 or 2
        in_features: the number of input features
        out_features: the number of output features
        groups: group in_features into "groups", "groups" argument of conv layer


        """
        super().__init__()
        self.layer_cfg = bank_cfg[layer_id] if bank_cfg is not None else None
        self.args = args
        self.in_channels = in_features
        self.out_channels = out_features
        self.num_shared_layers = num_shared_layers

        if set_input:
            template_in_size = in_features // groups
        else:
            template_in_size = max(1, (in_features * args.template_size[0]) // groups)

        if set_output:
            template_out_size = out_features
        else:
            template_out_size = int(out_features * args.template_size[0])

        self.num_in_combinations = int(np.ceil(float(in_features) / template_in_size))
        self.num_out_combinations = int(
            np.ceil(float(out_features) / template_out_size)
        )
        num_combinations = self.num_in_combinations * self.num_out_combinations

        ## Num templates to train 1 layer for the seed model
        self.coefficients = (
            nn.ParameterDict()
        )  ## coefficients[i] is the coefs of i-th ensemble member_id

        if bank_cfg is not None:
            num_combinations = self.layer_cfg["num_members"]
            for i in range(num_combinations):
                num_templates_i = self.layer_cfg[f"num_templates_{i}"]
                if set_input or set_output:  ## all coeffes are 1, no gradient
                    self.coefficients[f"coefs_member_{i}"] = nn.Parameter(
                        torch.ones(num_templates_i), requires_grad=False
                    )
                else:
                    self.coefficients[f"coefs_member_{i}"] = nn.Parameter(
                        torch.randn(num_templates_i)
                    )
        else:
            if len(set(args.num_templates_each_set)) == 1:
                num_templates = num_shared_layers * args.num_templates_each_set[0]
                coefs = torch.zeros([num_combinations, num_templates])
                nn.init.orthogonal_(coefs)

                for i in range(num_combinations):
                    self.coefficients[f"coefs_member_{i}"] = nn.Parameter(
                        coefs[i].clone()
                    )
            else:
                for i in range(num_combinations):
                    set_idx = self.args.member_template_sets[i]
                    self.coefficients[f"coefs_member_{i}"] = nn.Parameter(
                        torch.randn(args.num_templates_each_set[set_idx])
                    )

        self.current_num_combinations = 0

    def _add_templates_growing(self):
        """
        Add new templates when growing
        """
        num_extra_templates = self.args.add_extra_templates_growing

        # create a new self.coefficients whose size is greater than the old one, then copy the old one to the new one.
        new_coefficients = nn.ParameterDict()

        # get device from self.coefficients
        device = self.coefficients[f"coefs_member_0"].device

        new_trainable_parameters = nn.ParameterDict()
        for i in range(self.current_num_combinations):
            set_idx = self.args.member_template_sets[i]
            num_templates = self.coefficients[f"coefs_member_{i}"].data.shape[0]
            # self.args.num_templates_each_set[set_idx]
            total_templates = num_templates + num_extra_templates
            new_coefficients[f"coefs_member_{i}"] = nn.Parameter(
                torch.randn(total_templates, device=device)
            )  ## TODO: consider zero init
            new_coefficients[f"coefs_member_{i}"].data[:num_templates] = (
                self.coefficients[f"coefs_member_{i}"].data.detach().clone()
            )

            new_trainable_parameters[f"template_set_{set_idx}"] = nn.Parameter(
                torch.zeros(
                    [self.num_shared_layers * total_templates, *self.template_size],
                    device=device,
                )
            )
            nn.init.kaiming_normal_(new_trainable_parameters[f"template_set_{set_idx}"])

            new_trainable_parameters[f"template_set_{set_idx}"].data[
                : self.num_shared_layers * num_templates
            ] = (
                self.trainable_parameters[f"template_set_{set_idx}"]
                .data.detach()
                .clone()
            )

        self.coefficients = new_coefficients
        self.trainable_parameters = new_trainable_parameters

    def scale_coefs(self):
        eps = 1e-5
        coefs_1_norm = self.coefficients[f"coefs_member_1"].detach().norm()
        # for i in range(1, len(self.coefficients)):
        i = 0
        coef_scale = (
            self.coefficients[f"coefs_member_{i}"].detach().norm() + eps
        ) / coefs_1_norm
        self.coefficients[f"coefs_member_{i}"].data = (
            self.coefficients[f"coefs_member_{i}"].detach().data / coef_scale
        )

    def scale_templates(self):
        eps = 1e-6
        set_1_norm = self.trainable_parameters[f"template_set_1"].detach().norm()
        # for set_i in range(1, len(self.trainable_parameters)):
        set_i = 0
        template_scale = (
            self.trainable_parameters[f"template_set_{set_i}"].detach().norm() + eps
        ) / set_1_norm
        self.trainable_parameters[f"template_set_{set_i}"].data = (
            self.trainable_parameters[f"template_set_{set_i}"].detach().data
            / template_scale
        )

    def copy_templates_0_to_1(self):
        # random gaussian noise
        noise = self.args.copy_templates_and_bn_0_to_1_noise * torch.randn_like(
            self.trainable_parameters[f"template_set_0"]
        )

        self.trainable_parameters[f"template_set_1"].data = (
            self.trainable_parameters[f"template_set_0"].detach().data.clone() + noise
        )

    def growth_update(self, ind, evaluation=False):
        print("call growth_update")

        ## debug: TODO remove this
        print(
            "norm of template set 0",
            self.trainable_parameters[f"template_set_0"].detach().norm(),
        )
        print(
            "norm of template set 1",
            self.trainable_parameters[f"template_set_1"].detach().norm(),
        )

        if evaluation:
            return

        if ind == 0:
            return

        ## check on this for multiple growth
        if self.args.add_extra_templates_growing > 0:
            self._add_templates_growing()

        # ### Orthogonal init: from learned coefficients, generate new coefficients that are orthogonal to the old ones
        # # Special case for ensemble
        # if ind > self.old_num_combinations:
        #     old_end_idx = ind - 1
        # else:
        #     old_end_idx = self.old_num_combinations
        #
        # ## loop through all new coefficients, re-generating them
        # for i in range(old_end_idx, self.current_num_combinations - 1):
        #     breakpoint()
        #
        #     prev = [self.coefficients[f"coefs_member_{j}"].detach() for j in range(old_end_idx + i)]
        #     prev_coefficients = torch.stack(prev, dim=0)
        #     new_coefficients = torch.linalg.svd(prev_coefficients).Vh[-1]
        #     self.coefficients[f"coefs_member_{i + 1}"].data.copy_(new_coefficients)

        ## TODO: check this for multiple growth
        num_ensembles = len(self.args.template_size) - 1
        for i in range(num_ensembles, self.current_num_combinations):
            coefs_0 = self.coefficients[f"coefs_member_{0}"].data.detach().clone()
            coefs_1 = self.coefficients[f"coefs_member_{1}"].data.detach().clone()
            rand_coef = torch.rand_like(coefs_0)
            rand_coef /= rand_coef.norm()
            noise = self.args.coefs_noise_growing * rand_coef

            if self.args.coefs_growing == "copy":
                if self.args.member_template_sets == [0, 1, 0, 1]:
                    self.coefficients[f"coefs_member_{2}"].data = coefs_0 + noise
                    self.coefficients[f"coefs_member_{3}"].data = coefs_1 + noise
                    print(f"copy coefs from 0->2, 1->3")
                elif self.args.member_template_sets == [0, 1, 1, 0]:
                    self.coefficients[f"coefs_member_{2}"].data = coefs_1 + noise
                    self.coefficients[f"coefs_member_{3}"].data = coefs_0 + noise
                    print(f"copy coefs from 0->3, 1->2")
                else:
                    raise NotImplementedError(
                        f"member_template_sets {self.args.member_template_sets} not implemented"
                    )
                break
            elif self.args.coefs_growing == "orthogonal":
                if i == num_ensembles:
                    continue
                prev = [
                    self.coefficients[f"coefs_member_{j}"].detach().clone()
                    for j in range(i)
                ]
                prev_coefficients = torch.stack(prev, dim=0)
                new_coefficients = (
                    torch.linalg.svd(prev_coefficients).Vh[-1]
                    * self.args.coefs_noise_growing
                )
                self.coefficients[f"coefs_member_{i}"].data.copy_(new_coefficients)
                print(f"Orthogonal init for coefs_member_{i}")
            elif self.args.coefs_growing == "zero":
                self.coefficients[f"coefs_member_{i}"].data = torch.zeros_like(coefs_0)
                print(f"Zero init for coefs_member_{i}")
            elif self.args.coefs_growing == "random":
                # TODO: change later
                coefs_i = coefs_0 if i in [0, 2] else coefs_1
                rand_coef_i = torch.rand_like(coefs_i)
                # rand_coef_i /= rand_coef_i.norm()
                self.coefficients[f"coefs_member_{i}"].data = (
                    rand_coef_i * self.args.coefs_noise_growing
                )
                print(f"Random init for coefs_member_{i}")
            elif self.args.coefs_growing == "none":
                print(f"No reinit for coefs_member_{i}")
                pass
            else:
                raise NotImplementedError(
                    f"coefs_growing {self.args.coefs_growing} not implemented"
                )

        for i in range(self.current_num_combinations):
            print(
                f"norm of coefs {i} is {self.coefficients[f'coefs_member_{i}'].detach().norm()}"
            )

    def get_params(self, member_id: int | None) -> torch.Tensor:
        """
        member_id: the index of the member_id to get the parameters for, should be 0, 1, or None
        """

        if self.current_num_combinations > 1:
            assert member_id is None  ## fully grown model
            inds = torch.arange(self.current_num_combinations)
        elif member_id is not None:
            inds = [member_id]
        else:
            # all planned settings assume this setting doesn't come up
            raise NotImplementedError

        layer_weights = []
        # for each combination, get linear combination of templates using coefficients
        for ind in inds:
            set_i = self.args.member_template_sets[ind]
            templates = self.trainable_parameters[f"template_set_{set_i}"]
            num_extra_dims = templates.ndim - 1
            new_shape = [-1] + list(np.ones(num_extra_dims, np.int32))
            coefficients = self.coefficients[f"coefs_member_{ind}"].view(*new_shape)
            layer_weights.append(torch.sum(templates * coefficients, 0))
            # if self.args.debug:
            #     print(f"----------SModule: coefs_member_{ind}----------")
            #     print(f"----------template_set_{set_i}----------")

        # code assumes that the first two inds contain any pretrained ensemble members
        if len(layer_weights) == 1:
            out_weights = layer_weights[0]
        elif self.args.ensemble_growth_type == "diag":
            """
            assume
            layer_weights = torch.arange(current_num_combinations)
            and current_num_in_combinations = current_num_out_combinations
            with current_num_in_combinations in [2,3,4], we have out_weights as follows:

            tensor([[0, 2],
                    [3, 1]])

            tensor([[0, 2, 3],
                    [4, 5, 6],
                    [7, 8, 1]])

            tensor([[ 0,  2,  3,  4],
                    [ 5,  6,  7,  8],
                    [ 9, 10, 11, 12],
                    [13, 14, 15,  1]])


            In case num_in=3, num_out=2:
            tensor([[0, 2, 3],
                    [4, 5, 1]])

            In case num_in=1, num_out=2:
            tensor([[0],
                    [1]])

            num_in=2, num_out=1:
            tensor([[0, 1]])

            """
            out_weights = []
            current_ind = 1  # index of the last pretrained members
            for i in range(self.current_num_out_combinations):
                row_weights = []
                start_idx = (i * self.current_num_in_combinations) + current_ind
                remaining_weights = self.current_num_in_combinations
                if i == 0:
                    row_weights.append(layer_weights[0])
                    remaining_weights -= 1
                    # let's get to the first member_id after the members
                    start_idx += 1

                if i == (self.current_num_out_combinations - 1):
                    remaining_weights -= 1

                for j in range(start_idx, start_idx + remaining_weights):
                    row_weights.append(layer_weights[j])

                if i == (self.current_num_out_combinations - 1):
                    row_weights.append(layer_weights[1])

                out_weights.append(torch.hstack(row_weights))

            out_weights = torch.vstack(out_weights)

        else:
            raise NotImplementedError

        res = out_weights[: self.current_shape[0], : self.current_shape[1]]
        return res

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []
        for key, module in self._modules.items():
            if key == "bank":
                continue
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)

        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class SConv2d(SModule):
    def __init__(
        self,
        args,
        num_shared_layers,
        in_features,
        out_features,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=None,
        set_input=False,
        layer_id=None,
        bank=None,
        bank_cfg: Dict = None,
    ):
        super().__init__(
            args,
            num_shared_layers,
            in_features,
            out_features,
            groups,
            set_input,
            layer_id=layer_id,
            bank=bank,
            bank_cfg=bank_cfg,
        )

        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.groups = groups
        self.set_input = set_input
        assert in_features % groups == 0
        self.shape = [out_features, in_features // groups, kernel_size, kernel_size]
        if bank_cfg is not None:
            assert (
                self.shape == bank_cfg[layer_id]["full_shape"]
            ), f"Shape mismatch: {self.shape} vs {bank_cfg[layer_id]['full_shape']}"
        self.bias = bias

        if (
            set_input
        ):  ## First conv stem get all 3 channels, while other convs get only template_size (e.g., half)
            template_in_size = in_features // groups
        else:
            template_in_size = int(in_features * args.template_size[0]) // groups
        template_out_size = int(out_features * args.template_size[0])

        self.trainable_parameters = nn.ParameterDict()  # keys: set of templates.
        self.template_size = [
            template_out_size,
            template_in_size,
            kernel_size,
            kernel_size,
        ]

        ## args.num_template_sets e.g., [3,3]: 2 sets, each has 3 templates
        if bank_cfg is not None:
            if str(self.layer_cfg["group_id"]) in bank:
                for set_i in ["0", "1"]:
                    self.trainable_parameters[f"template_set_{set_i}"] = bank[
                        str(self.layer_cfg["group_id"])
                    ][set_i]
            else:
                bank[str(self.layer_cfg["group_id"])] = nn.ParameterDict()
                for set_i in ["0", "1"]:
                    bank[str(self.layer_cfg["group_id"])][set_i] = nn.Parameter(
                        torch.zeros(
                            [
                                num_shared_layers
                                * self.layer_cfg[f"num_templates_{set_i}"],
                                *self.template_size,
                            ]
                        )
                    )

                    self.trainable_parameters[f"template_set_{set_i}"] = bank[
                        str(self.layer_cfg["group_id"])
                    ][set_i]
                    nn.init.kaiming_normal_(
                        self.trainable_parameters[f"template_set_{set_i}"]
                    )
        else:
            for set_i, num_templates in enumerate(args.num_templates_each_set):
                self.trainable_parameters[f"template_set_{str(set_i)}"] = nn.Parameter(
                    torch.zeros(
                        [num_shared_layers * num_templates, *self.template_size]
                    )
                )
                nn.init.kaiming_normal_(
                    self.trainable_parameters[f"template_set_{str(set_i)}"]
                )

        self.growth_update(0)

    def track_template_grad(self, name):
        return lambda gradients: self.template_grad[name].append(
            torch.abs(gradients).sum().item()
        )

    def track_coef_grad(self, name):
        return lambda gradients: self.coef_grad[name].append(
            torch.abs(gradients).sum().item()
        )

    def growth_update(self, ind, evaluation=False):
        ## At first, num_combinations is set to 0,
        ## To start training, we update it to 1 for training the first member_id
        ## Then, we grow, and update it to 4 to train the whole network.

        out_features = int(self.shape[0] * self.args.template_size[ind])
        if self.set_input:
            in_features = self.shape[1] // self.groups
        else:
            in_features = (
                int(self.shape[1] * self.args.template_size[ind]) // self.groups
            )
        self.current_num_in_combinations = int(
            np.ceil(float(in_features) / self.template_size[1])
        )
        self.current_num_out_combinations = int(
            np.ceil(float(out_features) / self.template_size[0])
        )
        self.old_num_combinations = self.current_num_combinations
        self.current_num_combinations = (
            self.current_num_in_combinations * self.current_num_out_combinations
        )
        self.current_shape = [out_features, in_features, self.shape[2], self.shape[3]]

        super(SConv2d, self).growth_update(
            ind, evaluation
        )  ## orthogonal init for other pieces when growing

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, groups={groups}"
        )
        # if self.padding != (0,) * len(self.padding):
        #    s += ', padding={padding}'
        # if self.dilation != (1,) * len(self.dilation):
        #    s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #    s += ', output_padding={output_padding}'
        # if self.groups != 1:
        #    s += ', groups={groups}'
        # if self.bias is None:
        #    s += ', bias=False'
        # if self.padding_mode != 'zeros':
        #    s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def forward(self, x, member_id=None):
        params = self.get_params(member_id)
        # if self.args.debug:
        #     print("memberid", member_id)
        #     print("params", params.shape)
        #     print("params norm", params.norm())
        return F.conv2d(
            x,
            params,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=self.bias,
        )


class SLinear(SModule):
    def __init__(
        self,
        args,
        num_shared_layers,
        in_features,
        out_features,
        full_output=True,
        layer_id=None,
        bank=None,
        bank_cfg: Dict = None,
    ):
        super().__init__(
            args,
            num_shared_layers,
            in_features,
            out_features,
            set_output=full_output,
            layer_id=layer_id,
            bank=bank,
            bank_cfg=bank_cfg,
        )
        self.shape = [out_features, in_features]
        self.bias = None
        self.full_output = full_output

        template_in_size = int(in_features * args.template_size[0])
        if full_output:
            template_out_size = out_features
        else:
            template_out_size = int(out_features * args.template_size[0])

        self.trainable_parameters = nn.ParameterDict()  # keys: set of templates.
        self.template_size = [template_out_size, template_in_size]
        if bank_cfg is not None:
            if str(self.layer_cfg["group_id"]) in bank:
                for set_i in ["0", "1"]:
                    self.trainable_parameters[f"template_set_{set_i}"] = bank[
                        str(self.layer_cfg["group_id"])
                    ][set_i]
            else:
                bank[str(self.layer_cfg["group_id"])] = nn.ParameterDict()
                for set_i in ["0", "1"]:
                    bank[str(self.layer_cfg["group_id"])][set_i] = nn.Parameter(
                        torch.zeros(
                            [
                                num_shared_layers
                                * self.layer_cfg[f"num_templates_{set_i}"],
                                *self.template_size,
                            ]
                        )
                    )

                    self.trainable_parameters[f"template_set_{set_i}"] = bank[
                        str(self.layer_cfg["group_id"])
                    ][set_i]
                    nn.init.kaiming_normal_(
                        self.trainable_parameters[f"template_set_{set_i}"]
                    )
        else:
            for set_i, num_templates in enumerate(args.num_templates_each_set):
                self.trainable_parameters[f"template_set_{str(set_i)}"] = nn.Parameter(
                    torch.zeros(
                        [num_shared_layers * num_templates, *self.template_size]
                    )
                )
                nn.init.kaiming_normal_(
                    self.trainable_parameters[f"template_set_{str(set_i)}"]
                )

        self.growth_update(0)

    def growth_update(self, ind, evaluation=False):
        if self.full_output:
            out_features = self.shape[0]
        else:
            out_features = int(self.shape[0] * self.args.template_size[ind])

        in_features = int(self.shape[1] * self.args.template_size[ind])
        self.current_num_in_combinations = int(
            np.ceil(float(in_features) / self.template_size[1])
        )
        self.current_num_out_combinations = int(
            np.ceil(float(out_features) / self.template_size[0])
        )
        self.old_num_combinations = self.current_num_combinations
        self.current_num_combinations = (
            self.current_num_in_combinations * self.current_num_out_combinations
        )
        self.current_shape = [out_features, in_features]
        super(SLinear, self).growth_update(ind, evaluation)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def forward(self, x, member_id=None):
        params = self.get_params(member_id)
        return F.linear(x, params)


T = TypeVar("T", bound=nn.Module)


class Sequential(nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    _modules: Dict[str, nn.Module]  # type: ignore[assignment]

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: nn.Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> "Sequential":
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                "of Sequential class, but {} is given.".format(str(type(other)))
            )

    def pop(self, key: Union[int, slice]) -> nn.Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> "Sequential":
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError(
                "add operator supports only objects "
                "of Sequential class, but {} is given.".format(str(type(other)))
            )

    def __mul__(self, other: int) -> "Sequential":
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> "Sequential":
        return self.__mul__(other)

    def __imul__(self, other: int) -> "Sequential":
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input, member_id):
        for module in self:
            input = module(input, member_id)
        return input

    def append(self, module: nn.Module) -> "Sequential":
        r"""Appends a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: nn.Module) -> "Sequential":
        if not isinstance(module, nn.Module):
            raise AssertionError("module should be of type: {}".format(nn.Module))
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError("Index out of range: {}".format(index))
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> "Sequential":
        for layer in sequential:
            self.append(layer)
        return self
