# -*- coding = utf-8 -*-
# @Time : 2025/11/6 8:22 下午
# @Author Sun Jingchen
# @File predict.py
# @Software PyCharm
# -*- coding: utf-8 -*-
# version:2024.8.8
# 2024.8月以后竞赛必须使用该版本
# 项目名称/算法类型:目标检测
import os
import sys
import json
import numpy as np
import time
import subprocess

if "PaddleYOLO" not in sys.path:
    sys.path.append("PaddleYOLO")

import paddle
from ppdet.metrics import get_infer_results
from ppdet.core.workspace import load_config, create
from ppdet.engine import Trainer
from ppdet.data.source import get_categories


### 请在这里import需要调用的库


class Predictor:
    """
    InitModel函数  模型初始化参数,注意不能自行增加删除函数入参
    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    """

    def InitModel(self):
        ret = True
        err_message = "normal"
        """
        模型初始化,由用户自行编写
        加载出错时给ret和err_message赋值相应的错误
        *注意模型应为相对路径
        """
        ### 请在try-except内模型初始化代码,便于捕获错误
        try:
            self.cfg = load_config("configs/yolov10_s_100e_coco.yml")
            self.trainer = Trainer(self.cfg, mode="test")
            self.trainer.load_weights("output/yolov10_s_100e_coco/best_model.pdparams")
        except Exception as err:
            ret = False
            err_message = "[Error] 模型初始化错误,信息:[{}]".format(
                ExceptionMessage(err)
            )
            print(err_message)

        return ret, err_message

    """
    Detect         模型推理函数,注意不能自行增加删除函数入参
    file_path      单个待测图片路径,字符串类型
    return         列表[]
    """

    def Detect(self, file_path):
        ### 请在try-except内编写推理代码,便于捕获错误
        try:
            detect_result = []
            """
            [可选]读取图片,支持中文,与cv2.imread读取一致
            image_nparray=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)

            模型推理部分,由用户自行编写
            detect_result输出格式:
            [ 
                [category,xmin,ymin,xmax,ymax,score], # 目标0
                [category,xmin,ymin,xmax,ymax,score], # 目标1
                ....
                [category,xmin,ymin,xmax,ymax,score]  # 目标N
            ] 
            数据格式:                示例
            category : str          "person"      *类别名称必须与数据集中的类别名称一致,类别名称如"040100xx"
            xmin     : float/int    123.13
            ymin     : float/int    123.13
            xmax     : float/int    323.13
            ymax     : float/int    423.13
            score    : float/int    0.6666        目标置信度值
            用户根据输出格式append到detect_result
            """
            inference_dataset = create("TestDataset")()
            inference_dataset.set_images([file_path])
            loader = create("TestReader")(inference_dataset, 0)

            clsid2catid, catid2name = get_categories(
                "coco", inference_dataset.get_anno()
            )

            self.trainer.model.eval()
            with paddle.no_grad():
                for batch in loader:
                    output = self.trainer.model(batch)
                    output["im_id"] = batch["im_id"]
                    output = get_infer_results(output, clsid2catid)
                    for item in output["bbox"]:
                        catid = item["category_id"]
                        x, y, w, h = item["bbox"]
                        score = item["score"]
                        if score < 0.5:
                            continue
                        detect_result.append(
                            [catid2name[catid], x, y, x + w, y + h, score]
                        )

            return detect_result

        ### 请在try-except内编写推理代码,便于捕获错误
        except Exception as err:
            print("[Error] 模型推理错误.信息:{}".format(ExceptionMessage(err)))
            return err


### 获取异常文件+行号+信息
def ExceptionMessage(err):
    err_message = (
        str(err.__traceback__.tb_frame.f_globals["__file__"])
        + ":"
        + str(err.__traceback__.tb_lineno)
        + "行:"
        + str(err)
    )
    return err_message


### numpy数组转json辅助类
class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


if __name__ == "__main__":
    ###备注说明:main函数提供给用户内测,修改后[不影响]评估
    predictor = Predictor()
    ret, err_message = predictor.InitModel()
    if ret:

        ###读取遍历文件夹读取图片示例
        result_dict = {}
        test_dir = "D:/Projects/datasets/magic287/save_test"
        file_name_list = os.listdir(test_dir)
        for file_name in file_name_list:
            if file_name.lower().endswith(".xml"):
                continue
            start_timestamp = time.time()
            file_path = os.path.join(test_dir, file_name)
            detect_result = predictor.Detect(file_path)
            detect_time = round(time.time() - start_timestamp, 2)
            print(f"文件:{file_path}:推理结果:{detect_result}:耗时:{detect_time}秒")

            base_name = os.path.basename(file_name)
            result_dict[base_name] = detect_result

        """
        识别结果格式要求如下,{key:val}结构,其中key为文件名,value为识别结果;文件名强烈建议使用os.path.basename(file_name)
        {
            "test_00001.jpg": [["person",100,100,200,300,0.75],["bus",200,200,300,400,0.65]],
            "test_00002.jpg": [["dog",100,100,200,300,0.75],["cat",200,200,300,400,0.65]],
        }
        """

        # 识别结果result.json保存到文件夹save_result,相对路径!
        save_result_dir = "save_result"
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)

        # 保存识别结果到result.json
        with open(
            os.path.join(save_result_dir, "result.json"), "w", encoding="utf-8"
        ) as fw:
            json.dump(result_dict, fw, indent=4, ensure_ascii=False, cls=JsonEncoder)

        # 打包文件夹命令:
        subprocess.getstatusoutput(f"tar cf {save_result_dir}.tar {save_result_dir}")
    else:
        print(f"[Error] 模型初始化错误:{err_message}")
