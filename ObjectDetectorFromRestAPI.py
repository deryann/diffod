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


class ObjectDetectorFromRestAPI(ObjectDetector):
    """
    Add a interface for object detector RestAPI (for inference).
    You can imlement all function for your object detection model.
    """
    def load_model(self, dic_cfg={}):
        self.api_url = dic_cfg.get('api_url', None)
        self.model_name = dic_cfg.get('model_name', 'dummy')
        self.classes = None
        self.original_iou = 0.45
        self.original_conf = 0.25
        self.labels = self.get_labels()
        self.lock = threading.Lock()

        pass

    def clear_model(self):
        pass

    def get_supported_models(self):
        url = self.api_url+'/get_models'
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
        return t

    def get_labels(self):
        url = self.api_url+'/get_class'
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

    def inference_2_nparray_by_filepath(self, filename: str, dic_cfg=None):
        url = self.api_url+'/od_inference'
        if dic_cfg is not None:
            _iou, _conf = dic_cfg.get('iou_thres', self.original_iou), dic_cfg.get('conf_thres', self.original_conf)

        else:
            _iou, _conf = self.original_iou, self.original_conf

        payload = {
            "conf_thres": _conf,
            "iou_thres": _iou,
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
