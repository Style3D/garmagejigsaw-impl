import copy

import torch
import torch.nn as nn


def circular_pading(x, pad):
    if pad == 0:
        return x
    pad = min(pad, len(x))
    return torch.cat([x[-pad:], x, x[:pad]], dim=0)


def circular_clipping(x, pad):
    if pad == 0:
        return x
    return x[...,pad:-pad]


def cal_padding(kernel_size, dilation):
    effective_kernel = dilation * (kernel_size - 1) + 1  # = 5
    padding = (effective_kernel - 1) // 2
    return padding


class res_block_xd_default(nn.Module):
    """
    Copied from the HOLA B-REP offitial implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1, v_norm="layer", v_norm_shape=None):
        super(res_block_xd_default, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = 1
        self.v_norm = v_norm
        self.v_norm_shape = v_norm_shape

        self.downsample = None
        if v_norm is None or v_norm == "none":
            norm = nn.Identity()
        elif v_norm == "layer":
            raise NotImplementedError
            norm = nn.LayerNorm(v_norm_shape)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding
        )
        self.norm1 = copy.deepcopy(norm)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding
        )
        self.norm2 = copy.deepcopy(norm)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                copy.deepcopy(norm),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class feature_conv_layer_contourwise(nn.Module):
    """
    Mixes the features of points belonging to the same loop (contour).

    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param dilation:
    :param groups:
    :return:
    """
    def __init__(self, in_channels, out_channels, type="default", kernel_size=3,  dilation=1):
        super(feature_conv_layer_contourwise, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = 1
        self.type = type

        if self.type == "default":
            self.padding = (kernel_size - 1) // 2
            self.conv = res_block_xd_default(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = self.kernel_size,
                padding = self.padding,
                dilation = self.dilation,
                v_norm = None,
                v_norm_shape = None
            )

    def forward(self, x):
        if self.type == "default":
            x_padded = circular_pading(x, self.padding)
            x_padded = x_padded[None,...].transpose(1, 2)
            x_padded = self.conv(x_padded)
            x_clipped = circular_clipping(x_padded, self.padding)
            output = x_clipped.transpose(1, 2)[0]
            return output
        else:
            raise NotImplementedError