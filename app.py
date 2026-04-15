import streamlit as st
from inference import run_inference
from visualization import (
    create_mri_image,
    create_manual_segmentation,
    create_overlay_auto
)
from models import create_model
import numpy as np
import nibabel as nib


st.set_page_config(page_title="MRI Seg Viewer", layout="wide")
st.title("MRI Segmentation — Raw / Manual / Auto Overlay")


# -------------------------
# Upload Panel
# -------------------------
uploaded_mri = st.sidebar.file_uploader("MRI File (.nii)", type=["nii", "nii.gz"])
uploaded_manual = st.sidebar.file_uploader("Manual Segmentation (optional)", type=["nii"])
uploaded_weights = st.sidebar.file_uploader("Model weights (.pth)", type=["pth"])

# Save uploaded files
if uploaded_mri:
    with open("input_mri.nii", "wb") as f:
        f.write(uploaded_mri.read())
if uploaded_manual:
    with open("manual_seg.nii", "wb") as f:
        f.write(uploaded_manual.read())
if uploaded_weights:
    with open("weights.pth", "wb") as f:
        f.write(uploaded_weights.read())


# -------------------------
# RUN INFERENCE
# -------------------------
if uploaded_mri and uploaded_weights and st.sidebar.button("Run Segmentation"):
    mri, pred = run_inference(
        mri_path="input_mri.nii",
        model_path="weights.pth",
        model_builder=create_model
    )
    st.session_state.mri = mri
    st.session_state.pred = pred

    if uploaded_manual:
        manual = nib.load("manual_seg.nii")
        manual = nib.as_closest_canonical(manual).get_fdata()
        st.session_state.manual = manual


# -------------------------
# DISPLAY
# -------------------------
if "mri" in st.session_state:

    mri = st.session_state.mri
    pred = st.session_state.pred
    manual = st.session_state.get("manual", None)

    axis = st.selectbox("Axis", ["axial", "coronal", "sagittal"])

    if axis == "axial":
        max_idx = mri.shape[0] - 1
    elif axis == "coronal":
        max_idx = mri.shape[1] - 1
    else:
        max_idx = mri.shape[2] - 1

    idx = st.slider("Slice", 0, max_idx, max_idx // 2)


    # -------------------------
    # Generate Images
    # -------------------------
    img_raw = create_mri_image(mri, axis, idx)

    if manual is not None:
        img_manual = create_manual_segmentation(manual, axis, idx)
    else:
        img_manual = None

    img_overlay = create_overlay_auto(mri, pred, axis, idx)


    # -------------------------
    # Display in columns
    # -------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(img_raw, caption="MRI Raw")

    with c2:
        if img_manual:
            st.image(img_manual, caption="Manual Segmentation")
        else:
            st.info("Manual segmentation not uploaded.")

    with c3:
        st.image(img_overlay, caption="Auto Segmentation (Overlay)")