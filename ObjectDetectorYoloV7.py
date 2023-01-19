import os
import json
import traceback
from torch.hub import load
import numpy as np
import cv2
from PIL import Image
import torch
import gc
import threading
import requests
from ObjectDetector import ObjectDetector


class ObjectDetectorYoloV7(ObjectDetector):
    """
    Add a interface for object detector YOLOV5 (for inference).
    You can imlement all function for your object detection model.
    """

    def load_model(self, dic_cfg={}):

        # self.model = load('ultralytics/yolov5', dic_cfg.get('model_name', 'yolov5s'), pretrained=True)
        # self.labels = self.model.names
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        print('conf_thres', self.conf_thres)
        print('iou_thres', self.iou_thres)

        
        self.labels = self.get_labels()
        self.lock = threading.Lock()

        
        pass

    def clear_model(self):
        pass

    def get_labels(self):
        url = 'http://localhost:5000/get_class'
        try:
            response = requests.get(url)
            if response.ok:
                print(response.status_code)
                t = response.json()
            else:
                t = dict()
        except Exception as e:
            print(traceback.format_exc())
            t = dict()
        return t.get('class', [])

    def inference_2_nparray_by_filepath(self, filename: str):
        url = 'http://localhost:5000/od_inference'

        payload = {
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "model_name": self.model_name
            }

        files = {
            'data': (None, json.dumps(payload), 'application/json'),
            'file': (os.path.basename(filename), open(filename, 'rb'), 'application/octet-stream')
        }

        try:
            response = requests.post(url, files=files)
            if response.ok:
                t = response.json()
            else:
                print(f"[Error] You may not launch yolov7 api interface in {url}")
                t = dict()
        except Exception as e:
            print(traceback.format_exc())
            t = dict()
        pred = np.array(t.get('detections', []))

        return pred
