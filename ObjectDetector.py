from PIL import Image
import numpy as np
import cv2
import math


class ObjectDetector:
    """
    Add a interface for object detector (for inference).
    You can imlement all function for your object detection model.
    """
    color_list = [(29, 178, 255),
                  (168, 153, 44),
                  (49, 210, 207),
                  (243, 126, 162),
                  (89, 190, 22),
                  (207, 190, 23),
                  (99, 112, 171),
                  (194, 119, 227),
                  (180, 119, 31),
                  (40, 39, 214)]

    def __init__(self, dic_cfg={}) -> None:
        """
        initalize model
        """
        self.model = None
        self.labels = []
        self.model_name = dic_cfg.get('model_name', 'dummy') 
        self.load_model(dic_cfg)
        pass

    def load_model(self, dic_cfg={}):
        raise Exception(f"[WARNING] {__name__} is not implement")
        pass

    def clear_model():
        pass

    def inference_as_image_by_filepath(filename: str):
        raise Exception(f"[WARNING] {__name__} is not implement")

    def inference_as_image_by_filepath(self, filename: str):
        FONT_SCALE = 1.5* 1e-3  # Adjust for larger font size in all images
        FONT_THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
        LINE_THICKNESS_SCALE = 1e-2  # Adjust for larger thickness in all images
        TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box

        image = Image.open(filename)
        image = image.convert("RGB")
        image = np.array(image)
        height, width, _ = image.shape
        pred = self.inference_2_nparray_by_filepath(filename)
        for obj in pred:
            x1, y1, x2, y2, conf, label_idx = obj
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            _color = self.color_list[int(label_idx) % len(self.color_list)]
            font_thickness = math.ceil(min(width, height) * FONT_THICKNESS_SCALE)
            line_thickness = math.ceil(min(width, height) * LINE_THICKNESS_SCALE)
            cv2.rectangle(image, pt1, pt2, _color, line_thickness)
            _text = self.labels[label_idx] + f" {conf:.2f}"
            cv2.putText(image, _text, (pt1[0], pt1[1] - int(height * TEXT_Y_OFFSET_SCALE)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        min(width, height) * FONT_SCALE,
                        _color, font_thickness)

        # Convert image from numpy array to PIL image
        image = Image.fromarray(np.uint8(image))
        return image

    def inference_as_json_by_filepath(self, filename: str, dic_cfg = None):
        pred = self.inference_2_nparray_by_filepath(filename, dic_cfg)
        r = []
        for obj in pred:
            x1, y1, x2, y2, conf, label_idx = obj
            dic_r = {
                "x1": int(x1),
                "x2": int(x2),
                "y1": int(y1),
                "y2": int(y2),
                "conf": conf,
                "label": self.labels[int(label_idx)],
            }
            r.append(dic_r)
        return r

    def __del__(self):
        """
        clear your model 
        """

        self.clear_model()
        pass
