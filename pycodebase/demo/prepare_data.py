import os
import pandas as pd
import shutil

csv_path = "../data/raw/labels.csv"
image_dir = "../data/raw/images"
output_dir = "../data/train"

df = pd.read_csv(csv_path)

mapping = {
    "MEL": "melanoma",
    "NV": "nevus",
    "BCC": "bcc",
    "AK": "ak",
    "BKL": "bkl",
    "DF": "df",
    "VASC": "vasc",
    "SCC": "scc"
}

# OPTIONAL: speed up training
df = df.sample(3000)

for _, row in df.iterrows():
    img_name = row['image'] + ".jpg"

    label = None
    for col in mapping:
        if row[col] == 1:
            label = mapping[col]
            break

    if label is None:
        continue

    src = os.path.join(image_dir, img_name)
    dst_dir = os.path.join(output_dir, label)

    os.makedirs(dst_dir, exist_ok=True)

    dst = os.path.join(dst_dir, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("Dataset organized successfully!")