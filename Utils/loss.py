import tensorflow as tf
import warnings
from tensorflow.keras import backend as K


def categorical_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1):
    """Computes the categorical focal crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as mentioned in the reference.
        gamma: A focusing parameter, default is `2.0` as mentioned in the reference.
        from_logits: Whether `y_pred` is expected to be a logits tensor.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
        axis: Defaults to `-1`. The dimension along which the entropy is computed.

    Returns:
        Categorical focal crossentropy loss value.
    """
    if isinstance(axis, bool):
        raise ValueError(f"`axis` must be of type `int`. Received: axis={axis} of type {type(axis)}")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_focal_crossentropy, expected y_pred.shape to be (batch_size, num_classes) with num_classes > 1. "
            f"Received: y_pred.shape={y_pred.shape}. Consider using 'binary_crossentropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )

    if label_smoothing:
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=axis)

    # Adjust the predictions so that the probability of each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is computed correctly.
    output = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
    output = tf.clip_by_value(output, K.epsilon(), 1.0 - K.epsilon())

    # Calculate cross entropy
    cce = -y_true * tf.math.log(output)

    # Calculate factors
    modulating_factor = tf.math.pow(1.0 - output, gamma)
    weighting_factor = modulating_factor * alpha

    # Apply weighting factor
    focal_cce = weighting_factor * cce
    focal_cce = tf.reduce_sum(focal_cce, axis=axis)
    return focal_cce


def dice_loss(y_true, y_pred):
    """Computes the Dice loss value between `y_true` and `y_pred`.

    Formula:
    ```python
    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    ```

    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    Returns:
        Dice loss value.
    """
    y_true = tf.cast(y_true, y_pred.dtype)

    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice_coeff = (2.0 * intersection + K.epsilon()) / (
        tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + K.epsilon()
    )

    return 1 - dice_coeff


def ccfl_dice(y_true, y_pred, lamda=0.7):
    focal = categorical_focal_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return (lamda)*focal + (1-lamda)*dice