import traceback
from torch.hub import load
import numpy as np
import cv2
from PIL import Image
import torch
import gc

from ultralytics import YOLO
from ObjectDetector import ObjectDetector


class ObjectDetectorYoloV8(ObjectDetector):
    """
    Add a interface for object detector YOLOV8 (for inference).
    You can imlement all function for your object detection model.
    """

    def load_model(self, dic_cfg={}):
        self.model = YOLO(dic_cfg.get('model_name', 'yolov8s') + ".pt")  # load a pretrained model (recommended for training)
        self.labels = self.model.model.names
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

    def inference_2_nparray_by_filepath(self, filename: str):
        list_tensor = self.model(filename)

        if list_tensor is not None:
            pred = list_tensor[0].boxes.boxes.cpu().numpy()
        else:
            pred = np.array([])

        return pred
