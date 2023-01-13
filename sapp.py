
import sys
from streamlit.web import cli as stcli
from streamlit import runtime

import streamlit as st
from PIL import Image

import numpy as np
import cv2

from ObjectDetectorYoloV5 import ObjectDetectorYoloV5
from ObjectDetectorYoloV8 import ObjectDetectorYoloV8

CONST_MODEL_01 = 'yolov5s'
CONST_MODEL_02 = 'yolov8s'

d_yolov5 = ObjectDetectorYoloV5({'model_name':CONST_MODEL_01})
d_yolov8 = ObjectDetectorYoloV8({'model_name':CONST_MODEL_02})

def function_yolo_v8(image_file):
    return d_yolov8.inference_as_image_by_filepath(image_file)


def function_yolo_v5(image_file):
    return d_yolov5.inference_as_image_by_filepath(image_file)


def main():

    st.set_page_config(page_title="Diff Object Detection App", page_icon="figure/compare.png", layout="wide")

    st.title("Diff Object Detection App")

    # Create a file uploader widget
    image_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    if image_file is not None:
        # Open the image file
        fack_name = 'temp.jpg'
        image = Image.open(image_file)
        image.save(fack_name)

        with col1:
            st.header(f"YOLOV5 - {CONST_MODEL_01}")
            image = function_yolo_v5(fack_name)
            st.image(image, caption='V5 Result.', use_column_width=True)
        with col2:
            st.header(f"YOLOV8 - {CONST_MODEL_02}")
            image = function_yolo_v8(fack_name)
            st.image(image, caption='V8 Result.', use_column_width=True)


if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
