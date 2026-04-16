# 🧠 MRI Segmentation with SwinUNet (PyTorch)

A complete deep learning pipeline for  medical image segmentation using the **SwinUNet architecture** implemented in PyTorch.

---

## 📌 Overview

This project provides an **end-to-end framework** for MRI segmentation, including:

* Data preprocessing & normalization
* SwinUNet model architecture
* Training with checkpointing
* Evaluation (Dice, IoU, etc.)
* Inference on new MRI scans
* Visualization tools
* Optional dataset organization (e.g., BraTS)
* Interactive web application

---

## 📁 Project Structure

```
MRI_Segmentation_SwinUNet/
│
├── main.py
├── train.py
├── train_utils.py
├── models.py
├── evaluation.py
├── inference.py
├── preprocessing.py
├── visualization.py
├── organize_brats.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MRI_Segmentation_SwinUNet.git

cd MRI_Segmentation_SwinUNet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

```bash
python organize_brats.py \
  --input_dir path/to/raw_dataset \
  --output_dir data/processed
```

Structure:

```
data/
├── train/
├── val/
└── test/
```

---

## 🧠 Training

```bash
python train.py \
  --data_dir data/train \
  --val_dir data/val \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4 \
  --model swinunet

## 🌐 Web App

```bash
streamlit run app.py
```

Features:

* Upload MRI
* Run segmentation
* Visualize results

---

## 🧩 Model Architecture

* Swin Transformer encoder
* U-Net style decoder
* Skip connections
* Window-based self-attention

---

## ⚙️ Requirements

* PyTorch
* NumPy
* nibabel / SimpleITK
* scikit-learn
* Plotly / Streamlit

---

## 📜 License

MIT License
