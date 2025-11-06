import json
import random
import argparse

from glob import glob
import os.path as osp
import xml.etree.ElementTree as ET


def voc_to_coco(xml_names, labels, annotations_path, images_path, output_json_path):
    images = []
    annotations = []
    categories = [{"id": i + 1, "name": label} for i, label in enumerate(labels)]

    for xml_name in xml_names:
        with open(osp.join(annotations_path, xml_name), "r", encoding="utf-8") as f:
            root = ET.fromstring(f.read())
        image_names = glob(images_path + "/" + osp.splitext(xml_name)[0] + ".*")
        image_names = [
            osp.basename(n) for n in image_names if not n.lower().endswith(".xml")
        ]
        assert len(image_names) == 1
        image_name = image_names[0]
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        image_id = len(images)
        images.append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_name,
            }
        )

        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = max(min(float(bbox.find("xmin").text), width), 1)
            xmax = max(min(float(bbox.find("xmax").text), width), 1)
            ymin = max(min(float(bbox.find("ymin").text), height), 1)
            ymax = max(min(float(bbox.find("ymax").text), height), 1)
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                    "area": w * h,
                    "category_id": labels.index(label) + 1,
                    "ignore": 0,
                }
            )

    with open(output_json_path, "w") as f:
        json.dump(
            {
                "categories": categories,
                "info": {},
                "images": images,
                "annotations": annotations,
            },
            f,
            indent=4,
        )


def get_labels(annotations_path):
    labels = set()
    xml_paths = glob(annotations_path + "/*.xml")
    for xml_path in xml_paths:
        root = ET.parse(xml_path)
        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.add(label)
    return sorted(list(labels))


def train_val_split(annotations_path, train_ratio):
    xml_paths = glob(annotations_path + "/*.xml")
    xml_filenames = [osp.basename(xml_path) for xml_path in xml_paths]
    indices = list(range(len(xml_filenames)))
    random.seed(86)
    random.shuffle(indices)
    train_size = int(len(xml_filenames) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_xml_filenames = [xml_filenames[i] for i in train_indices]
    val_xml_filenames = [xml_filenames[i] for i in val_indices]
    return train_xml_filenames, val_xml_filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-path", type=str)
    parser.add_argument("--images-path", type=str)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    labels = get_labels(args.annotations_path)
    train_xmls, val_xmls = train_val_split(args.annotations_path, args.train_ratio)
    voc_to_coco(
        train_xmls,
        labels,
        args.annotations_path,
        args.images_path,
        osp.join(args.output, "train.json"),
    )
    voc_to_coco(
        val_xmls,
        labels,
        args.annotations_path,
        args.images_path,
        osp.join(args.output, "val.json"),
    )
