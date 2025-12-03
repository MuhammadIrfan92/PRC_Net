import torch
import torch.nn as nn
import torch.nn.functional as F


def global_covariance_pooling(x: torch.Tensor) -> torch.Tensor:
    """
    Compute global covariance pooling.

    Expects input x of shape (B, C, H, W) (channels-first).
    Returns covariance matrix of shape (B, C, C).
    """
    B, C, H, W = x.shape
    N = H * W
    # flatten spatial dims: (B, C, N)
    x_flat = x.view(B, C, N)
    mean = x_flat.mean(dim=2, keepdim=True)  # (B, C, 1)
    centered = x_flat - mean  # (B, C, N)
    # covariance: (B, C, C) = centered @ centered^T / N
    cov = torch.bmm(centered, centered.transpose(1, 2)) / float(N)
    return cov


class GlobalCovPoolingLayer(nn.Module):
    """Module wrapper for global covariance pooling (channels-first)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return global_covariance_pooling(x)


class ChannelSpatialAttention(nn.Module):
    """
    Channel-spatial attention (CSA) converted to a PyTorch-friendly SE-like implementation.

    Input: (B, C, H, W)
    Output: (B, C, H, W) multiplied by learned channel attention (per-channel scalar).
    """

    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.channels = channels
        mid = max(1, channels // ratio)
        # gcp processing: conv1d over the covariance matrix sequence dimension
        self.gcp_conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        # replace Linear-based fc with Conv1d-based equivalent that operates on (B, C, 1)
        # first conv maps C -> mid (kernel_size=1), second maps mid -> C
        self.fc_conv1 = nn.Conv1d(in_channels=channels, out_channels=mid, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc_conv2 = nn.Conv1d(in_channels=mid, out_channels=channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # channel descriptors
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)   # (B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)   # (B, C)

        # global covariance pooling -> (B, C, C)
        gcp = global_covariance_pooling(x)                  # (B, C, C)
        # treat gcp as (B, channels, seq_len) for conv1d -> out (B, channels, seq_len)
        gcp_processed = self.gcp_conv1(gcp)                # (B, C, C)
        # global average over seq dim -> (B, C)
        gcp_vec = gcp_processed.mean(dim=2)

        # combined descriptor and excitation
        combined = avg_pool + max_pool + gcp_vec            # (B, C)

        # use Conv1d-based fc: reshape to (B, C, 1) and apply convs
        combined_seq = combined.unsqueeze(2)               # (B, C, 1)
        out = self.fc_conv1(combined_seq)                  # (B, mid, 1)
        out = self.relu(out)
        out = self.fc_conv2(out)                           # (B, C, 1)
        scale = self.sigmoid(out).view(B, C, 1, 1)         # (B, C, 1, 1)
        return x * scale