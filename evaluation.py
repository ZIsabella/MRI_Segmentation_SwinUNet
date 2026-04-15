import numpy as np
import scipy.ndimage


def _resize_to_match(gt, ref_shape):
    zoom = (
        ref_shape[0] / gt.shape[0],
        ref_shape[1] / gt.shape[1],
        ref_shape[2] / gt.shape[2],
    )
    return scipy.ndimage.zoom(gt, zoom, order=0)


def dice_score(pred, gt):
    if gt is None:
        return None

    if pred.shape != gt.shape:
        gt = _resize_to_match(gt, pred.shape)

    pred = pred > 0.5
    gt = gt > 0.5

    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter) / (pred.sum() + gt.sum() + 1e-8)


def iou_score(pred, gt):
    if gt is None:
        return None

    if pred.shape != gt.shape:
        gt = _resize_to_match(gt, pred.shape)

    pred = pred > 0.5
    gt = gt > 0.5

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return inter / (union + 1e-8)