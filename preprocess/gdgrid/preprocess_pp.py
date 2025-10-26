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
                label_ids[label] = len(label_ids) + 1
    categories = [{"id": i, "name": label} for label, i in label_ids.items()]

    indices = list(range(len(df)))
    random.seed(10086)
    random.shuffle(indices)
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    resized_images_folder = osp.join(dataset_path, "resized_images")
    os.makedirs(resized_images_folder, exist_ok=True)
    for split, split_indices in {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }.items():
        images = []
        annotations = []
        for i in tqdm(split_indices, desc=split):
            annotation = json.loads(df.iloc[i][5])
            image_path = df.iloc[i][4]
            image_name = osp.basename(image_path)
            image = cv2.imread(osp.join(dataset_path, image_path))
            height, width, _ = image.shape
            ratio = min(1, new_edge / min(height, width))
            new_height = int(height * ratio)
            new_width = int(width * ratio)
            image = cv2.resize(image, (new_width, new_height))
            cv2.imwrite(osp.join(resized_images_folder, image_name), image)

            images.append(
                {
                    "id": i,
                    "width": new_width,
                    "height": new_height,
                    "file_name": image_name,
                }
            )

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
                        "image_id": i,
                        "category_id": label_ids[label],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )

        with open(osp.join(dataset_path, f"{split}.json"), "w") as f:
            json.dump(
                {
                    "info": info,
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
                f,
            )

    dataset_yml_content = f"""metric: COCO
num_classes: {len(label_ids)}

TrainDataset:
  name: COCODataSet
  image_dir: {resized_images_folder}
  anno_path: {osp.join(dataset_path, "train.json")}
  dataset_dir: {dataset_path}
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  name: COCODataSet
  image_dir: {resized_images_folder}
  anno_path: {osp.join(dataset_path, "val.json")}
  dataset_dir: {dataset_path}

TestDataset:
  name: ImageFolder
  anno_path: {osp.join(dataset_path, "test.json")}
  dataset_dir: {dataset_path}
"""

    with open("configs-dataset/gdgrid_detection.yml", "w") as f:
        f.write(dataset_yml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GuangDong grid detection dataset preprocess for PaddlePaddle"
    )
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.dataset_path, "3train_rname.csv"), header=None)
    train_val_test_split(args.dataset_path, df, 0.8, 0.1, 1024)
