import numpy as np
from PIL import Image


def normalize_slice(slice_2d):
    mn = np.min(slice_2d)
    mx = np.max(slice_2d)
    return (slice_2d - mn) / (mx - mn + 1e-8)


def mri_slice(mri, axis, index):
    if axis == "axial":
        return mri[index, :, :].T
    elif axis == "coronal":
        return mri[:, index, :]
    elif axis == "sagittal":
        return mri[:, :, index]
    else:
        raise ValueError("Invalid axis")


def seg_slice(seg, axis, index):
    if axis == "axial":
        return seg[index, :, :].T
    elif axis == "coronal":
        return seg[:, index, :]
    elif axis == "sagittal":
        return seg[:, :, index]
    else:
        raise ValueError("Invalid axis")


# -----------------------------------------------------------
# 1) MRI RAW IMAGE
# -----------------------------------------------------------
def create_mri_image(mri, axis, index):
    sl = mri_slice(mri, axis, index)
    sl_norm = normalize_slice(sl)
    img = (np.stack([sl_norm] * 3, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(img)


# -----------------------------------------------------------
# 2) MANUAL SEGMENTATION IMAGE
# -----------------------------------------------------------
def create_manual_segmentation(seg, axis, index):
    sl = seg_slice(seg, axis, index)
    mask = sl > 0.5
    mask_img = np.zeros((sl.shape[0], sl.shape[1], 3), dtype=np.uint8)

    # رنگ سبز برای segmentation دستی
    mask_img[mask] = [0, 255, 0]

    return Image.fromarray(mask_img)


# -----------------------------------------------------------
# 3) OVERLAY (MRI + AUTO SEG)
# -----------------------------------------------------------
def create_overlay_auto(mri, auto_seg, axis, index):
    sl_mri = mri_slice(mri, axis, index)
    sl_pred = seg_slice(auto_seg, axis, index)

    mri_norm = normalize_slice(sl_mri)
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)

    mask = sl_pred > 0.5
    overlay = mri_rgb.copy()

    # قرمز برای سگمنت مدل
    overlay[mask] = [1.0, 0.0, 0.0]

    overlay_img = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay_img)
