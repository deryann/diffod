import copy
import sys
from streamlit.web import cli as stcli
from streamlit import runtime

import streamlit as st
from PIL import Image

import numpy as np
import math
import cv2

from ObjectDetectorYoloV5 import ObjectDetectorYoloV5
from ObjectDetectorYoloV8 import ObjectDetectorYoloV8

from compare_utility import *
CONST_MODEL_01 = 'yolov5s'
CONST_MODEL_02 = 'yolov8s'

d_yolov5 = ObjectDetectorYoloV5({'model_name':CONST_MODEL_01})
d_yolov8 = ObjectDetectorYoloV8({'model_name':CONST_MODEL_02})

def get_image_compare_result (filename:str, lst_base, lst_to, color_same=(0,255,0) , color_diff= (255,0,0)):
    FONT_SCALE = 1.5* 1e-3  # Adjust for larger font size in all images
    FONT_THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
    LINE_THICKNESS_SCALE = 1e-2  # Adjust for larger thickness in all images
    TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box

    image = Image.open(filename)
    image = image.convert("RGB")
    image = np.array(image)
    height, width, _ = image.shape
    lst_results = get_iou_compare_result(lst_base, lst_to)
    for obj in lst_results:
        x1, y1, x2, y2, conf, label = obj['x1'], obj['y1'], obj['x2'], obj['y2'], obj['conf'], obj['label']
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        if obj.get('link', None) is None:
            _color = color_diff
        else:
            _color = color_same
        font_thickness = math.ceil(min(width, height) * FONT_THICKNESS_SCALE)
        line_thickness = math.ceil(min(width, height) * LINE_THICKNESS_SCALE)
        cv2.rectangle(image, pt1, pt2, _color, line_thickness)
        _text = label + f" {conf:.2f}"
        cv2.putText(image, _text, (pt1[0], pt1[1] - int(height * TEXT_Y_OFFSET_SCALE)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    min(width, height) * FONT_SCALE,
                    _color, font_thickness)

    # Convert image from numpy array to PIL image
    image = Image.fromarray(np.uint8(image))
    return image

def main():

    st.set_page_config(page_title="Diff Object Detection", page_icon="figure/compare.png", layout="wide")

    st.title("Diff Object Detection")

    # Create a file uploader widget
    image_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    if image_file is not None:
        # Open the image file
        _temp_name = 'temp.jpg'
        image = Image.open(image_file)
        image.save(_temp_name)
        
        lst_v5_result = d_yolov5.inference_as_json_by_filepath(_temp_name)
        lst_v8_result = d_yolov8.inference_as_json_by_filepath(_temp_name)

        with col1:
            st.header(f"YOLOV5 - {CONST_MODEL_01}")
            image_1 = get_image_compare_result(_temp_name, copy.deepcopy(lst_v5_result), copy.deepcopy(lst_v8_result))
            st.image(image_1, caption='V5 Result.', use_column_width=True)
        with col2:
            st.header(f"YOLOV8 - {CONST_MODEL_02}")
            image_2 = get_image_compare_result(_temp_name, lst_v8_result, lst_v5_result, color_diff=(0,0,255))
            st.image(image_2, caption='V8 Result.', use_column_width=True)


if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
