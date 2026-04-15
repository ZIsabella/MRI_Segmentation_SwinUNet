import os
import shutil

src_root = "braTSMiniDataset"
dst_img = "Task01_BrainTumour/imagesTr"
dst_lbl = "Task01_BrainTumour/labelsTr"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

cases = sorted(os.listdir(src_root))
idx = 1

for case in cases:
    case_path = os.path.join(src_root, case)

    if not os.path.isdir(case_path):
        continue

    files = os.listdir(case_path)

    img_path = None
    seg_path = None

    for f in files:
        f_lower = f.lower()

        # 🔥 MRI (FLAIR / t2f)
        if "t2f" in f_lower:
            img_path = os.path.join(case_path, f)

        # 🔥 label
        elif "seg" in f_lower:
            seg_path = os.path.join(case_path, f)

    if img_path and seg_path:

        # ❌ قبلاً اشتباه بود: .nii.gz
        # ✔ درست: همان فرمت اصلی (.nii)
        ext = os.path.splitext(img_path)[1]   # .nii یا .nii.gz

        name = f"case_{idx:03d}{ext}"

        shutil.copy(img_path, os.path.join(dst_img, name))
        shutil.copy(seg_path, os.path.join(dst_lbl, name))

        print(f"Processed: {case}")
        idx += 1

print("Done!")