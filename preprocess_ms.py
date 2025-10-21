import os
import os.path as osp
import argparse
import json
import random

import pandas as pd
import cv2
from tqdm import tqdm


def train_val_test_split(dataset_path, df, train_ratio, val_ratio, new_edge):
    label_replace = {"监护袖章(红only)": "badge"}

    label_ids = {}
    for i in range(len(df)):
        annotation = json.loads(df.iloc[i][5])
        for item in annotation["items"]:
            label = item["labels"]["标签"]
            label = label_replace.get(label, label)
            if label_ids.get(label) is None:
                label_ids[label] = len(label_ids)

    indices = list(range(len(df)))
    random.seed(10086)
    random.shuffle(indices)
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    ms_folder = osp.join(dataset_path, "ms_coco")
    for split, split_indices in {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }.items():
        os.makedirs(osp.join(ms_folder, "images", split), exist_ok=True)
        os.makedirs(osp.join(ms_folder, "labels", split), exist_ok=True)

        with open(osp.join(ms_folder, f"{split}.txt"), "w") as f:
            for index in range(len(split_indices)):
                f.write(
                    osp.abspath(
                        osp.join(ms_folder, "images", split, f"{index:012}.jpg")
                    )
                )

        for index, i in enumerate(tqdm(split_indices, desc=split)):
            annotation = json.loads(df.iloc[i][5])
            image_path = df.iloc[i][4]
            image = cv2.imread(osp.join(dataset_path, image_path))
            height, width, _ = image.shape
            ratio = min(1, new_edge / min(height, width))
            new_height = int(height * ratio)
            new_width = int(width * ratio)
            image = cv2.resize(image, (new_width, new_height))
            cv2.imwrite(osp.join(ms_folder, "images", split, f"{index:012}.jpg"), image)

            f = open(osp.join(ms_folder, "labels", split, f"{index:012}.txt"), "w")
            for item in annotation["items"]:
                label = item["labels"]["标签"]
                label = label_replace.get(label, label)
                bbox = item["meta"]["geometry"]
                bbox[0] = int(min(max(bbox[0], 0), width - 1) * ratio)
                bbox[1] = int(min(max(bbox[1], 0), height - 1) * ratio)
                bbox[2] = int(min(max(bbox[2], 0), width - 1) * ratio)
                bbox[3] = int(min(max(bbox[3], 0), height - 1) * ratio)

                x, y = bbox[:2]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x, y, w, h = x / width, y / height, w / width, h / height
                label_id = label_ids[label]
                f.write(f"{label_id} {x} {y} {w} {h}\n")
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GuangDong grid detection dataset preprocess for MindSpore"
    )
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.dataset_path, "3train_rname.csv"), header=None)
    train_val_test_split(args.dataset_path, df, 0.8, 0.1, 1024)
