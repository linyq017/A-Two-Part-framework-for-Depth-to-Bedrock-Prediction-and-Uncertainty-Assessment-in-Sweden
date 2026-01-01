"""Prediction fusion."""
import numpy as np


def fuse_predictions(binary_proba, qrf_predictions, binary_threshold=0.5):
    """Fuse binary and QRF predictions."""
    is_outcrop = binary_proba >= binary_threshold
    fused_predictions = qrf_predictions.copy()
    fused_predictions[is_outcrop, :] = 0
    
    return {
        'fused_predictions': fused_predictions,
        'is_outcrop': is_outcrop,
        'outcrop_proba': binary_proba
    }