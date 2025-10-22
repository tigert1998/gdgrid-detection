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
    info = {
        "description": "GuangDong grid detection dataset",
        "url": "https://tianchi.aliyun.com/competition/entrance/531897/information",
    }

    label_ids = {}
    for i in range(len(df)):
        annotation = json.loads(df.iloc[i][5])
        for item in annotation["items"]:
            label = item["labels"]["标签"]
            label = label_replace.get(label, label)
            if label_ids.get(label) is None:
                label_ids[label] = len(label_ids)
    categories = [{"id": i, "name": label} for label, i in label_ids.items()]

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
        "train2017": train_indices,
        "val2017": val_indices,
        "test2017": test_indices,
    }.items():
        images = []
        annotations = []

        f = open(osp.join(ms_folder, f"{split}.txt"), "w")
        for index in range(len(split_indices)):
            f.write(
                osp.realpath(osp.join(ms_folder, "images", split, f"{index:012}.jpg"))
                + "\n"
            )
        f.close()

        os.makedirs(osp.join(ms_folder, "images", split), exist_ok=True)
        os.makedirs(osp.join(ms_folder, "labels", split), exist_ok=True)
        os.makedirs(osp.join(ms_folder, "annotations"), exist_ok=True)

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

            images.append(
                {
                    "id": index,
                    "width": new_width,
                    "height": new_height,
                    "file_name": f"{index:012}.jpg",
                }
            )

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
                annotations.append(
                    {
                        "id": len(annotations),
                        "image_id": index,
                        "category_id": label_ids[label],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )

                x = (x + w / 2) / new_width
                y = (y + h / 2) / new_height
                w = w / new_width
                h = h / new_height
                f.write(f"{label_ids[label]} {x} {y} {w} {h}\n")
            f.close()

        with open(
            osp.join(ms_folder, "annotations", f"instances_{split}.json"), "w"
        ) as f:
            json.dump(
                {
                    "info": info,
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
                f,
            )

    label_names_array_str = (
        "[" + ", ".join(map(lambda s: f'"{s}"', label_ids.keys())) + "]"
    )
    dataset_yml_content = f"""data:
  dataset_name: gdgrid_detection

  train_set: {osp.realpath(osp.join(ms_folder, "train2017.txt"))}
  val_set: {osp.realpath(osp.join(ms_folder, "val2017.txt"))}

  nc: {len(label_ids)}

  names: {label_names_array_str}

  train_transforms: []
  test_transforms: []
"""

    with open("configs/gdgrid_detection_ms.yml", "w") as f:
        f.write(dataset_yml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GuangDong grid detection dataset preprocess for MindSpore"
    )
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.dataset_path, "3train_rname.csv"), header=None)
    train_val_test_split(args.dataset_path, df, 0.8, 0.1, 1024)
