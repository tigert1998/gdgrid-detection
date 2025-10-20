# -*- coding: utf-8 -*-
# version:2024.8.8
# 2024.8月以后竞赛必须使用该版本
# 项目名称:输电线路山火灾害智能识别

import sys
import os
import paddle
import paddle.nn as nn
import numpy as np
from tqdm import tqdm
import typing
import cv2
from PIL import Image, ImageOps, ImageFile
import time

if "PaddleYOLO" not in sys.path:
    sys.path.append("PaddleYOLO")

from ppdet.core.workspace import load_config, create
from ppdet.engine import Trainer
from ppdet.data.source.category import get_categories
from ppdet.metrics import get_infer_results
from ppdet.utils.logger import setup_logger

# from ppdet.modeling.initializer import reset_initialized_parameter

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = setup_logger("ppdet.engine")


class Predictor:

    # InitModel函数 模型初始化参数，注意不能自行增加删除函数入参
    # Params:
    # ret: 是否正常: 正常True,异常False
    # err_message: 错误信息: 默认normal
    # return ret, err_message

    def __init__(self):
        self.trainer = None
        self.cfg = None

    def InitModel(self):
        ret = True
        err_message = "normal"

        # 模型初始化，由用户自行编写
        # 加载出错时给ret和err_message赋相应的值的错误
        # *注意模型应为相对路径

        try:
            # 输入配置文件和模型文件路径
            config_path = "configs/yolov10_s_100e_coco.yml"  # 根据目录
            model_path = "output/yolov10_s_100e_coco/best_model.pdparams"  # 根据目录

            # 读取配置文件
            if os.path.exists(config_path):
                self.cfg = load_config(config_path)
            else:
                return False, "配置文件读取失败，请检查文件是否正确"

            self.trainer = Trainer(self.cfg, mode="test")

            # 读取模型文件
            if os.path.exists(model_path):
                self.trainer.load_weights(model_path)
            elif os.path.exists(self.cfg.weights):
                self.trainer.load_weights(self.cfg.weights)
            else:
                return False, "模型文件读取失败，请检查文件是否正确"

            # 检查是否使用xpu进行预测服务
            # 默认不使用xpu进行预测，可在configs里进行配置
            if "use_xpu" not in self.cfg:
                self.cfg.use_xpu = False
            if self.cfg.use_xpu:
                paddle.set_device("xpu")  # 设置device('xpu')

        except Exception as err:
            ret = False
            err_message = "[Error] model init failed, err_message:{}".format(
                ExceptionMessage(err)
            )
            print(err_message)
            return ret, err_message

        return ret, err_message

    # Detect
    # 模型推理，注意不能自行增加删除函数入参
    # 单张待预测图片路径，字符串类型
    # return: [[]]

    def Detect(self, file_path):
        ### 请在try内编写推理代码，便于捕获错误
        try:
            """
            模型推理部分，由用户自行编写
            detect_result输出格式:
            [
                [category, xmin, ymin, xmax, ymax, score],
                [category, xmin, ymin, xmax, ymax, score],
                ....
                [category, xmin, ymin, xmax, ymax, score]
            ]
            数据格式:               示例
            category:       str         dog * 类别名称必须与数据集中的类别名称一致
            xmin    :       float       123.13
            ymin    :       float       123.13
            xmax    :       float       323.13
            ymax    :       float       423.13
            score   :       float       0.6666
            用户根据输出格式append到detect_result
            """

            images = [file_path]
            detect_result = self._predict_nw(images)
            return detect_result

        except Exception as err:
            print(
                "[Error] predictor.Detect failed. err_message:{}".format(
                    ExceptionMessage(err)
                )
            )
            return err

    # 基于PaddleYOLOv5库系列的推理函数示例
    def _predict_nw(self, images):
        self.trainer.dataset.set_images(images)
        loader = create("TestReader")(self.trainer.dataset, 0)
        imid2path = self.trainer.dataset.get_imid2path()

        # 读取coco标注信息，主要包含类别，及该类别对应的类别ID
        anno_file = self.trainer.dataset.get_anno()
        clsid2catid, catid2name = get_categories(self.cfg.metric, anno_file=anno_file)
        print("[INFO] catid2name = ", catid2name)

        # Run Infer
        self.trainer.status["mode"] = "test"
        self.trainer.model.eval()
        if self.cfg.get("print_flops", False):
            flops_loader = create("TestReader")(self.trainer.dataset, 0)
            self.trainer.flops(flops_loader)

        # 保存模型输出结果
        results = []
        test_batch_size = self.cfg.TestReader["batch_size"]
        logger.info(
            "Test loader length is ({}), test batch size is ({}).".format(
                len(loader), test_batch_size
            )
        )
        logger.info("Starting predicting ......\n")

        for step_id, data in enumerate(tqdm(loader)):
            self.trainer.status["step_id"] = step_id
            # forward
            outs = self.trainer.model(data)

            for key in ["im_shape", "scale_factor", "im_id"]:
                if isinstance(data[key], typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]

            for key, value in outs.items():
                if hasattr(value, "numpy"):
                    outs[key] = value.numpy()

            results.append(outs)

        # 将【模型输出结果】转换成【两网定制的标注输出结果】
        # img = cv2.imread("./dataset/test2.png") # 示例
        infer_results = []
        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs["bbox_num"]

            start = 0
            for i, im_id in enumerate(outs["im_id"]):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert("RGB")
                image = ImageOps.exif_transpose(image)
                self.trainer.status["original_image"] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res["bbox"][start:end]
                start = end

                # 筛选 visualize results函数中的内容
                for dt in np.array(bbox_res):
                    if int(im_id) != dt["image_id"]:
                        continue
                    catid, bbox, score = dt["category_id"], dt["bbox"], dt["score"]

                    if score >= 0.5:
                        if len(bbox) == 4:
                            # draw bbox
                            # print(bbox)
                            xmin, ymin, w, h = bbox
                            xmin = max(xmin, 0)
                            ymin = max(ymin, 0)
                            w = max(w, 0)
                            h = max(h, 0)

                            xmax = xmin + w
                            ymax = ymin + h

                            # 加入infer_results中
                            # cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 10) # 示例
                            infer_results.append(
                                [catid2name[catid], xmin, ymin, xmax, ymax, score]
                            )

                        elif len(bbox) == 8:
                            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
                            xmin = min(x1, x2, x3, x4)
                            ymin = min(y1, y2, y3, y4)
                            xmax = max(x1, x2, x3, x4)
                            ymax = max(y1, y2, y3, y4)

                            # 加入infer_results中
                            infer_results.append(
                                [catid2name[catid], xmin, ymin, xmax, ymax, score]
                            )
                        else:
                            logger.error("The shape of bbox must be [M, 4] or [M, 8]!")

                # cv2.imwrite('output/test1287.jpg', img) # 示例

        return infer_results


# 获取异常文件行号+信息
def ExceptionMessage(err):
    err_message = (
        str(err.__traceback__.tb_frame.f_globals["__file__"])
        + ":"
        + str(err.__traceback__.tb_lineno)
        + "行:"
        + str(err)
    )

    return err_message


if __name__ == "__main__":
    # ### 仅说明:train.py会自动调用用户内的InitModel,修改后不影响评估
    predictor = Predictor()
    ret, err_message = predictor.InitModel()
    if ret:
        ### 读取单张图片示例
        start_timestamp = time.time()
        file_path = "D:/Projects/datasets/gdgrid/3_images/00a5337f_6bdb_4492_99fe_e2a1a06f3ac7.jpg"
        detect_result = predictor.Detect(file_path)
        detect_time = round(time.time() - start_timestamp, 2)
        print(f"文件:{file_path}:推理结果:{detect_result}:耗时:{detect_time}秒")

    ###读取遍历文件夹读取图片示例
    # test_dir=os.path.join("mydataset","Images")
    # file_name_list=os.listdir(test_dir)
    # for file_name in file_name_list:
    #     start_timestamp=time.time()
    #     file_path=os.path.join(test_dir,file_name)
    #     detect_result=predictor.Detect(file_path)
    #     detect_time=round(time.time()-start_timestamp,2)
    #     print(f"文件:{file_path}:推理结果:{detect_result}:耗时:{detect_time}秒")
    else:
        print("[Error] InitModel failed.ret", ret, err_message)
