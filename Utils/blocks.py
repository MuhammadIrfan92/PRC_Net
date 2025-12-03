import torch
import torch.nn as nn
import torch.nn.functional as F

from .attentions import ChannelSpatialAttention


class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise separable conv block:
      depthwise conv (groups=in_channels) -> GELU -> pointwise conv -> GELU
    Expects channels-first tensors.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.act(x)
        return x


class LSFB(nn.Module):
    """
    Local Separable Feature Block (stacked depthwise separable convs with concatenation).
    This mirrors the TensorFlow LSFB which concatenates intermediate outputs.
    """

    def __init__(self, in_channels: int, filters: int, kernel_size: int, num_layers: int,
                 stride: int = 1, dilation: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        # build a ModuleList where each layer expects the current channels and outputs `filters`
        self.layers = nn.ModuleList()
        cur_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DepthwiseSeparableConvBlock(cur_channels, filters, kernel_size, stride, dilation))
            cur_channels = cur_channels + filters  # will be used after concatenation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat_list = [x]
        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.dropout(out)
            concat_list.append(out)
            out = torch.cat(concat_list, dim=1)  # channel concat
        return out


class PRF(nn.Module):
    """
    Parallel Receptive Field (inception-like) block.
    Implements branches concatenated progressively and a final 1x1 conv to combine.
    """

    def __init__(self, in_channels: int, filters: int = 32, dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # branch1
        self.b1_conv = nn.Conv2d(in_channels, filters, kernel_size=1, padding=0)
        # branch2 (note: input channels is in_channels + filters after concat)
        self.b2_conv = nn.Conv2d(in_channels + filters, filters, kernel_size=3, padding=1)
        # branch3 (input channels grows again)
        self.b3_conv = nn.Conv2d(in_channels + 2 * filters, filters, kernel_size=5, padding=2)
        # final combine conv to produce `filters` output channels
        self.out_conv = nn.Conv2d(in_channels + 3 * filters, filters, kernel_size=1, padding=0)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.act(self.b1_conv(x))
        b1 = self.dropout(b1)
        b1_out = torch.cat([x, b1], dim=1)

        b2 = self.act(self.b2_conv(b1_out))
        b2 = self.dropout(b2)
        b2_out = torch.cat([b1_out, b2], dim=1)

        b3 = self.act(self.b3_conv(b2_out))
        b3 = self.dropout(b3)
        b3_out = torch.cat([b2_out, b3], dim=1)

        out = self.act(self.out_conv(b3_out))
        return out


class CCB(nn.Module):
    """
    Cascaded Atrous Convolution Block (CCB).
    Applies atrous/dilated convolutions, concatenates and reduces channels, then applies channel attention.
    """

    def __init__(self, in_channels: int, filters: int = 32):
        super().__init__()
        self.atrous1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, dilation=1)
        self.atrous2 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=2, dilation=2)
        self.atrous3 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=3, dilation=3)
        # reduce channels back to filters
        self.reduce = nn.Conv2d(filters * 3, filters, kernel_size=1)
        # channel-spatial attention
        self.att = ChannelSpatialAttention(filters)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.atrous1(x)
        a2 = self.atrous2(x)
        a3 = self.atrous3(x)
        concat = torch.cat([a1, a2, a3], dim=1)
        reduced = self.reduce(concat)
        attended = self.att(reduced)
        return self.act(attended)


class PRF_atten(nn.Module):
    """
    PRF followed by CSA attention and ReLU activation.
    """

    def __init__(self, in_channels: int, filters: int, dropout_rate: float = 0.0):
        super().__init__()
        self.prf = PRF(in_channels, filters, dropout_rate=dropout_rate)
        self.att = ChannelSpatialAttention(filters)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prf(x)
        x = self.att(x)
        x = self.act(x)
        return x