import os.path as osp
import argparse
import json
import random

import pandas as pd
import cv2
from tqdm import tqdm


def train_val_test_split(dataset_path, df, train_ratio, val_ratio):
    label_ids = {}
    for i in range(len(df)):
        annotation = json.loads(df.iloc[i][5])
        for item in annotation["items"]:
            label = item["labels"]["标签"]
            if label_ids.get(label) is None:
                label_ids[label] = len(label_ids)
    categories = [{"id": i, "name": label} for label, i in label_ids.items()]

    indices = list(range(len(df)))
    random.shuffle(indices)
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    image_folder = None
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
            image = cv2.imread(osp.join(args.dataset_path, image_path))
            height, width, _ = image.shape
            images.append(
                {
                    "id": i,
                    "width": width,
                    "height": height,
                    "file_name": osp.basename(image_path),
                }
            )
            if image_folder is None:
                image_folder = osp.dirname(image_path)
            else:
                assert image_folder == osp.dirname(image_path)

            for item in annotation["items"]:
                bbox = item["meta"]["geometry"]
                x, y = bbox[:2]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                annotations.append(
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": label_ids[label],
                        "bbox": [x, y, w, h],
                    }
                )
        with open(osp.join(dataset_path, f"{split}.json"), "w") as f:
            json.dump(
                {
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
  image_dir: {image_folder}
  anno_path: {osp.join(dataset_path, "train.json")}
  dataset_dir: {dataset_path}
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  name: COCODataSet
  image_dir: {image_folder}
  anno_path: {osp.join(dataset_path, "val.json")}
  dataset_dir: {dataset_path}

TestDataset:
  name: ImageFolder
  anno_path: {osp.join(dataset_path, "test.json")}
  dataset_dir: {dataset_path}
"""

    with open("configs/gdgrid_detection.yml", "w") as f:
        f.write(dataset_yml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GuangDong grid detection dataset preprocess")
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.dataset_path, "3train_rname.csv"), header=None)
    train_val_test_split(args.dataset_path, df, 0.8, 0.1)
