# ...existing code...
import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.blocks import PRF_atten, CCB, LSFB


class Conv2dBlock(nn.Module):
    """Two 3x3 convs with ReLU and optional Dropout (matches TF pattern)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.act(self.conv2(x))
        return x


class PRC_Net(nn.Module):
    def __init__(self, n_classes=3, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1, dropout_rate=0.0, l2_reg=0.0):
        """
        PyTorch port of the original Keras PRC_Net.
        Expects inputs in (B, C, H, W) with C == IMG_CHANNELS.
        Returns softmax probabilities over channels dim (dim=1).
        """
        super().__init__()

        block1_filters = 24
        block2_filters = 48
        block3_filters = 128
        block4_filters = 256
        bottleneck_filters = 256
        lsfb_layers = 4

        # Encoder: PRF_atten blocks
        self.c1 = PRF_atten(IMG_CHANNELS, block1_filters, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = PRF_atten(block1_filters, block2_filters, dropout_rate=dropout_rate)
        self.c3 = PRF_atten(block2_filters, block3_filters, dropout_rate=dropout_rate)
        self.c4 = PRF_atten(block3_filters, block4_filters, dropout_rate=dropout_rate)

        # c4skip via CCB
        self.c4skip_block = CCB(block4_filters, filters=block4_filters)

        # Bottleneck LSFB
        self.lsfb = LSFB(block4_filters, bottleneck_filters, kernel_size=3, num_layers=lsfb_layers, dropout_rate=dropout_rate)
        # compute channels output by LSFB: in + num_layers * filters
        c5_in_channels = block4_filters + lsfb_layers * bottleneck_filters

        # Decoder (transpose convs + conv blocks)
        self.up6 = nn.ConvTranspose2d(c5_in_channels, block4_filters, kernel_size=2, stride=2)
        self.conv6 = Conv2dBlock(block4_filters + block4_filters, block4_filters, dropout=dropout_rate)

        self.up7 = nn.ConvTranspose2d(block4_filters, block3_filters, kernel_size=2, stride=2)
        self.conv7 = Conv2dBlock(block3_filters + block3_filters, block3_filters, dropout=dropout_rate)

        self.up8 = nn.ConvTranspose2d(block3_filters, block2_filters, kernel_size=2, stride=2)
        self.conv8 = Conv2dBlock(block2_filters + block2_filters, block2_filters, dropout=dropout_rate)

        self.up9 = nn.ConvTranspose2d(block2_filters, block1_filters, kernel_size=2, stride=2)
        self.conv9 = Conv2dBlock(block1_filters + block1_filters, block1_filters, dropout=dropout_rate)

        # final 1x1 conv to classes
        self.out_conv = nn.Conv2d(block1_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        c1 = self.c1(x)            # -> block1_filters
        p1 = self.pool(c1)

        c2 = self.c2(p1)           # -> block2_filters
        p2 = self.pool(c2)

        c3 = self.c3(p2)           # -> block3_filters
        p3 = self.pool(c3)

        c4 = self.c4(p3)           # -> block4_filters
        p4 = self.pool(c4)

        c4skip = self.c4skip_block(c4)  # used for skip connection

        c5 = self.lsfb(p4)         # shape: (B, c5_in_channels, H/16, W/16)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4skip], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        outputs = self.out_conv(c9)
        probs = F.softmax(outputs, dim=1)
        return probs
# ...existing code...