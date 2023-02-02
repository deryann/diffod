import traceback
from torch.hub import load
import numpy as np
import cv2
from PIL import Image
import torch
import gc
from ObjectDetector import ObjectDetector


class ObjectDetectorYoloV5(ObjectDetector):
    """
    Add a interface for object detector YOLOV5 (for inference).
    You can imlement all function for your object detection model.
    """

    def load_model(self, dic_cfg={}):

        self.model = load('ultralytics/yolov5', dic_cfg.get('model_name', 'yolov5s'), pretrained=True)
        self.labels = self.model.names
        self.original_iou = self.model.iou
        self.original_conf = self.model.conf
        pass

    def clear_model(self):
        if self.model is not None:
            try:
                del self.model
                torch.cuda.memory_cached()
                torch.cuda.empty_cache()
                gc.collect()
                self.model = None
            except Exception:
                print("fail to delete model")
                print(traceback.format_exc())
        pass

    def inference_2_nparray_by_filepath(self, filename: str, dic_cfg=None):
        if dic_cfg is not None:
            self.model.iou = dic_cfg.get('iou_thres', self.original_iou)
            self.model.conf = dic_cfg.get('conf_thres', self.original_conf)
        else:
            self.model.iou = self.original_iou
            self.model.conf = self.original_conf

        image = Image.open(filename)
        image = image.convert("RGB")
        image = np.array(image)
        results = self.model(image)
        pred = results.xyxy[0].detach().cpu().numpy()
        return pred
