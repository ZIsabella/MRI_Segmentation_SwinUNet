import torch
import numpy as np
import nibabel as nib
import scipy.ndimage


def preprocess_mri(nifti_path, out_shape=(96, 96, 96), device="cpu"):
    """
    Load NIfTI from path and return tensor (1,1,D,H,W)
    """

    img = nib.load(nifti_path).get_fdata().astype(np.float32)

    # normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # resize
    img = scipy.ndimage.zoom(
        img,
        (
            out_shape[0] / img.shape[0],
            out_shape[1] / img.shape[1],
            out_shape[2] / img.shape[2],
        ),
        order=1
    )

    # (D,H,W) → (1,1,D,H,W)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    img = img.to(device)

    return img, None
