from PIL import Image
import numpy as np
import cv2


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
        image = Image.open(filename)
        image = image.convert("RGB")
        image = np.array(image)
        pred = self.inference_2_nparray_by_filepath(filename)
        for obj in pred:
            x1, y1, x2, y2, conf, label_idx = obj
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            _color = self.color_list[int(label_idx) % len(self.color_list)]
            cv2.rectangle(image, pt1, pt2, _color, 2)
            _text = self.labels[label_idx] + f" {conf:.2f}"
            cv2.putText(image, _text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, _color, 2)

        # Convert image from numpy array to PIL image
        image = Image.fromarray(np.uint8(image))
        return image

    def inference_as_json_by_filepath(self, filename: str):
        pred = self.inference_2_nparray_by_filepath(filename)
        r = []
        for obj in pred:
            x1, y1, x2, y2, conf, label_idx = obj
            dic_r = {
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "conf": conf,
                "label": self.labels[label_idx],
            }
            r.append(dic_r)
        return r

    def __del__(self):
        """
        clear your model 
        """

        self.clear_model()
        pass