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


class Predictor:
    def __init__(self):
        self.trainer = None
        self.cfg = None

    def init_model(self, config_path, model_path):
        self.cfg = load_config(config_path)
        self.trainer = Trainer(self.cfg, mode="test")
        self.trainer.load_weights(model_path)

    def predict(self, image_paths):
        inference_dataset = create("TestDataset")()
        inference_dataset.set_images(image_paths)
        loader = create("TestReader")(inference_dataset, 0)

        anno_file = self.trainer.dataset.get_anno()
        clsid2catid, catid2name = get_categories(self.cfg.metric, anno_file=anno_file)

        self.trainer.model.eval()

        ret = []
        for data in tqdm(loader):
            with paddle.no_grad():
                out = self.trainer.model(data)
            out["im_id"] = data["im_id"]
            infer_results = get_infer_results(out, clsid2catid)
            im_id = data["im_id"].item()

            outs = []
            for infer_result in infer_results["bbox"]:
                assert im_id == infer_result["image_id"]
                catid, bbox, score = (
                    infer_result["category_id"],
                    infer_result["bbox"],
                    infer_result["score"],
                )
                assert len(bbox) == 4

                if score >= 0.5:
                    x, y, w, h = bbox
                    outs.append([catid2name[catid], x, y, x + w, y + h, score])
            ret.append(outs)

        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predictor")
    parser.add_argument("--config")
    parser.add_argument("--params")
    parser.add_argument("--image")
    args = parser.parse_args()

    predictor = Predictor()
    predictor.init_model(args.config, args.params)
    predict_results = predictor.predict([args.image])
    print(predict_results[0])
