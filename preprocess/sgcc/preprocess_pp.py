import os
import os.path as osp
import xml.etree.ElementTree as ET
import random
import json
import argparse
from glob import glob

import numpy as np
import cv2
from tqdm import tqdm


def resize_image(image, edge):
    height, width, _ = image.shape
    ratio = min(1, edge / min(height, width))
    new_height, new_width = int(height * ratio), int(width * ratio)
    new_image = cv2.resize(image, (new_width, new_height))
    return new_image, ratio, new_width, new_height


def train_val_test_split(
    dataset_path,
    train_ratio,
    val_ratio,
):
    resized_images_folder = osp.join(dataset_path, "resized_images")
    os.makedirs(resized_images_folder, exist_ok=True)

    info = {"description": "SGCC dataset"}
    label_ids = {}

    pbar = tqdm(desc="Resizing images")
    instances = []
    for dirpath, dirnames, _ in os.walk(dataset_path):
        if "Annotation" in dirnames and "JPEGImage" in dirnames:
            annotation_dir = osp.join(dirpath, "Annotation")
            for xml_filename in os.listdir(annotation_dir):
                if not xml_filename.lower().endswith(".xml"):
                    continue
                xml_path = osp.join(annotation_dir, xml_filename)
                image_paths = glob(f"{dirpath}/JPEGImage/{xml_filename[:-4]}.*")
                assert len(image_paths) == 1
                image_path = image_paths[0]

                image = cv2.imdecode(
                    np.fromfile(file=image_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                image, ratio, width, height = resize_image(image, 1024)
                image_id = len(instances)
                new_image_name = f"{image_id:012}.jpg"
                new_image_path = osp.join(
                    dataset_path, "resized_images", new_image_name
                )
                cv2.imencode(".jpg", image)[1].tofile(new_image_path)
                instances.append((xml_path, new_image_path, ratio, width, height))

                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    label = obj.find("name").text.strip()
                    if label_ids.get(label) is None:
                        label_ids[label] = len(label_ids) + 1
                pbar.update(1)
    pbar.close()

    categories = [
        {"name": label, "id": label_id} for label, label_id in label_ids.items()
    ]

    random.seed(10086)
    indices = list(range(len(instances)))
    random.shuffle(indices)
    train_size = int(len(instances) * train_ratio)
    val_size = int(len(instances) * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    for split, split_indices in {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }.items():
        images = []
        annotations = []

        for index in tqdm(split_indices, desc=split):
            ratio, width, height = instances[index][2:5]

            images.append(
                {
                    "id": index,
                    "width": width,
                    "height": height,
                    "file_name": osp.basename(instances[index][1]),
                }
            )

            tree = ET.parse(instances[index][0])
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name").text.strip()
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                x = xmin * ratio
                y = ymin * ratio
                w = (xmax - xmin) * ratio
                h = (ymax - ymin) * ratio
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
                if label in ["aqd_gfsy", "aqd_dggy"]:
                    annotations.append(
                        {
                            "id": len(annotations),
                            "image_id": index,
                            "category_id": label_ids["aqd_zqpd"],
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                elif label in ["aqd_zqpd"]:
                    annotations.append(
                        {
                            "id": len(annotations),
                            "image_id": index,
                            "category_id": label_ids["aqd_gfsy"],
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )

        with open(osp.join(dataset_path, f"{split}.json"), "w") as f:
            json.dump(
                {
                    "info": info,
                    "categories": categories,
                    "images": images,
                    "annotations": annotations,
                },
                f,
                indent=4,
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

    with open("configs-dataset/sgcc_detection.yml", "w") as f:
        f.write(dataset_yml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SGCC detection dataset preprocess for PaddlePaddle"
    )
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()
    train_val_test_split(args.dataset_path, 0.95, 0.05)
