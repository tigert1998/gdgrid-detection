import sys
import argparse

from tqdm import tqdm

if "PaddleYOLO" not in sys.path:
    sys.path.append("PaddleYOLO")

import paddle
from ppdet.core.workspace import load_config, create
from ppdet.engine import Trainer
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results

from ensemble_boxes import weighted_boxes_fusion


class Predictor:
    def __init__(self):
        self.trainer = []
        self.cfg = []
        self.weights = []

    def init_model(self, config_path, model_path, weights):
        for i in range(len(config_path)):
            self.cfg.append(load_config(config_path[i]))
            self.trainer.append(Trainer(self.cfg[-1], mode="test"))
            self.trainer[-1].load_weights(model_path[i])
        self.weights = weights

    def predict(self, image_paths):
        inference_dataset = create("TestDataset")()
        inference_dataset.set_images(image_paths)
        loader = create("TestReader")(inference_dataset, 0)

        anno_file = inference_dataset.get_anno()
        clsid2catid, catid2name = get_categories("coco", anno_file=anno_file)

        returns = []
        for data in tqdm(loader):
            _, _, height, width = data["image"].shape

            boxes = []
            scores = []
            labels = []
            for trainer in self.trainer:
                trainer.model.eval()
                with paddle.no_grad():
                    output = trainer.model(data)
                output["im_id"] = data["im_id"]
                output = get_infer_results(output, clsid2catid)

                b = []
                s = []
                l = []
                for item in output["bbox"]:
                    catid = item["category_id"]
                    x, y, w, h = item["bbox"]
                    score = item["score"]
                    if score < 0.5:
                        continue
                    b.append([x / width, y / height, (x + w) / width, (y + h) / height])
                    s.append(score)
                    l.append(catid)

                boxes.append(b)
                scores.append(s)
                labels.append(l)

            boxes, scores, labels = weighted_boxes_fusion(
                boxes, scores, labels, self.weights
            )
            returns.append(
                [
                    [
                        catid2name[labels[i]],
                        float(boxes[i][0]) * width,
                        float(boxes[i][1]) * height,
                        float(boxes[i][2]) * width,
                        float(boxes[i][3]) * height,
                        float(scores[i]),
                    ]
                    for i in range(len(boxes))
                ]
            )

        return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predictor")
    parser.add_argument("--config", nargs="+")
    parser.add_argument("--params", nargs="+")
    parser.add_argument("--weights", nargs="+", default=[1], type=float)
    parser.add_argument("--image")
    args = parser.parse_args()

    predictor = Predictor()
    predictor.init_model(args.config, args.params, args.weights)
    predict_results = predictor.predict([args.image])
    print(predict_results[0])
