import warnings
import torch
import torch.nn.functional as F


def categorical_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1):
    """PyTorch implementation of categorical focal crossentropy.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets (logits or probabilities).
        alpha: Weighting factor for classes.
        gamma: Focusing parameter.
        from_logits: If True, apply softmax to y_pred.
        label_smoothing: Float in [0,1] for label smoothing.
        axis: Dimension of classes (default -1).

    Returns:
        Tensor of per-sample focal crossentropy losses (reduced along `axis`).
    """
    if isinstance(axis, bool):
        raise ValueError(f"`axis` must be of type `int`. Received: axis={axis} of type {type(axis)}")

    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true, dtype=y_pred.dtype)

    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_focal_crossentropy, expected y_pred.shape to be (batch_size, num_classes) with num_classes > 1. "
            f"Received: y_pred.shape={tuple(y_pred.shape)}. Consider using 'binary_cross_entropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )

    if label_smoothing:
        num_classes = float(y_true.shape[-1])
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    if from_logits:
        y_pred = F.softmax(y_pred, dim=axis)

    # normalize to ensure probabilities sum to 1 along classes
    denom = torch.sum(y_pred, dim=axis, keepdim=True)
    # avoid division by zero
    denom = torch.where(denom == 0, torch.tensor(1.0, dtype=denom.dtype, device=denom.device), denom)
    output = y_pred / denom

    eps = torch.finfo(output.dtype).eps
    output = torch.clamp(output, min=eps, max=1.0 - eps)

    # cross entropy per class
    cce = -y_true * torch.log(output)

    # focal factors
    modulating_factor = torch.pow(1.0 - output, gamma)
    weighting_factor = modulating_factor * alpha

    focal_cce = weighting_factor * cce
    focal_cce = torch.sum(focal_cce, dim=axis)  # reduce over classes
    return focal_cce


def dice_loss(y_true, y_pred, eps: float = 1e-7):
    """Dice loss. Flattens inputs and computes a single scalar loss across the entire tensor.

    Args:
        y_true: Tensor of ground truth (same shape as y_pred).
        y_pred: Tensor of predictions (probabilities or logits — apply appropriate activation before calling if needed).
        eps: small constant to avoid zero division.

    Returns:
        Scalar tensor: 1 - dice_coefficient
    """
    y_true = torch.as_tensor(y_true, dtype=y_pred.dtype)
    y_pred = torch.as_tensor(y_pred, dtype=y_pred.dtype)

    y_true_flat = y_true.contiguous().view(-1)
    y_pred_flat = y_pred.contiguous().view(-1)

    intersection = torch.sum(y_true_flat * y_pred_flat)
    denom = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    dice_coeff = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice_coeff


def ccfl_dice(y_true, y_pred, lamda=0.7, **focal_kwargs):
    """Combined categorical focal crossentropy + dice loss.

    Returns a per-sample tensor: lamda * focal + (1-lamda) * dice (dice is broadcasted).
    """
    focal = categorical_focal_crossentropy(y_true, y_pred, **focal_kwargs)
    dice = dice_loss(y_true, y_pred)
    return lamda * focal + (1.0 - lamda) * dice