import torch
from monai.networks.nets import SwinUNETR


def create_model():
    """
    ساخت مدل SwinUNETR سازگار با وزن‌های (48,96,192,384,768)
    """
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=1,
        feature_size=48,     # بسیار مهم: مطابق وزن‌های شما
        use_checkpoint=False
    )

    return model
