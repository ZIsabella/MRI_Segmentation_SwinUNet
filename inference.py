import torch
import numpy as np
import nibabel as nib
import scipy.ndimage


def load_mri_ras(path):
    """
    MRI را لود کرده و به فضای استاندارد RAS تبدیل می‌کند.
    """
    nii = nib.load(path)
    ras_nii = nib.as_closest_canonical(nii)
    data = ras_nii.get_fdata().astype(np.float32)
    return data, ras_nii.affine


def preprocess_mri(mri):
    """
    ورودی MRI به مدل SwinUNETR — خروجی shape = (1,1,96,96,96)
    """
    mri_norm = (mri - mri.mean()) / (mri.std() + 1e-8)
    resized = scipy.ndimage.zoom(
        mri_norm,
        (
            96 / mri.shape[0],
            96 / mri.shape[1],
            96 / mri.shape[2],
        ),
        order=1,
    )
    return resized[np.newaxis, np.newaxis, ...].astype(np.float32)


def resize_prediction(pred, original_mri):
    """
    خروجی مدل (96³) را به اندازه MRI اصلی برمی‌گرداند.
    """
    scale = (
        original_mri.shape[0] / pred.shape[0],
        original_mri.shape[1] / pred.shape[1],
        original_mri.shape[2] / pred.shape[2],
    )
    return scipy.ndimage.zoom(pred, scale, order=0)


def run_inference(mri_path, model_path, model_builder):
    """
    این تابع تمام مراحل inference را کامل انجام می‌دهد:
    - load MRI
    - preprocess
    - build model
    - load weights
    - predict
    - resize back
    """

    mri, affine = load_mri_ras(mri_path)
    model_input = preprocess_mri(mri)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_builder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(torch.tensor(model_input).to(device))
        output_np = output.cpu().numpy()[0, 0]

    pred_resized = resize_prediction(output_np, mri)

    return mri, pred_resized
