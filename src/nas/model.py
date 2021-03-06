from typing import List

import torch

import ops
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import basic_unit


@basic_unit
class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                nn.LayerChoice([
                    ops.PoolBN('max', channels, 3, stride, 1, affine=False),
                    ops.PoolBN('avg', channels, 3, stride, 1, affine=False),
                    nn.Identity() if stride == 1 else ops.FactorizedReduce(channels, channels, affine=False),
                    ops.SepConv(channels, channels, 3, stride, 1, affine=False),
                    ops.SepConv(channels, channels, 5, stride, 2, affine=False),
                    ops.DilConv(channels, channels, 3, stride, 2, 2, affine=False),
                    ops.DilConv(channels, channels, 5, stride, 4, 2, affine=False)
                ]))
        self.drop_path = ops.DropPath()
        self.input_switch = nn.InputChoice(n_candidates=num_prev_nodes, n_chosen=2)

    def forward(self, prev_nodes: List['Tensor']) -> 'Tensor':
        # assert self.ops.__len__() == len(prev_nodes)
        # out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = []
        for i, op in enumerate(self.ops):
            out.append(op(prev_nodes[i]))
        # out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)


class Cell(nn.Module):

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0))

    def forward(self, s0, s1):
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        new_tensors = []
        for node in self.mutable_ops:
            tmp = tensors + new_tensors
            cur_tensor = node(tmp)
            new_tensors.append(cur_tensor)

        output = torch.cat(new_tensors, dim=1)
        return output


class CNN(nn.Module):

    def __init__(self, input_size, in_channels, channels, n_classes, n_layers, n_nodes=4,
                 stem_multiplier=3, auxiliary=False):
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur)
        )

        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)

        return logits

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p


if __name__ == '__main__':
    base_model = CNN(32, 3, 16, 10, 8)
