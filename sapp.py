
import sys
from streamlit.web import cli as stcli
from streamlit import runtime

import streamlit as st
from PIL import Image
from torch.hub import load
import numpy as np
import cv2

from ultralytics import YOLO


CONST_MODEL_01 = 'yolov5s'
model_v5 = load('ultralytics/yolov5', CONST_MODEL_01, pretrained=True)

CONST_MODEL_02 = 'yolov8s'
model_v8 = YOLO(CONST_MODEL_02 + ".pt")  # load a pretrained model (recommended for training)

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


def function_yolo_v8(image_file):
    fack_name = 'temp.jpg'
    image = Image.open(image_file)
    image.save(fack_name)
    img = np.array(image)
    list_tensor = model_v8(fack_name)  # predict on an image
    pred = list_tensor[0].detach().cpu().numpy()
    for obj in pred:
        x1, y1, x2, y2, conf, label_idx = obj
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        _color = color_list[int(label_idx) % len(color_list)]
        cv2.rectangle(img, pt1, pt2, _color, 2)
        _text = model_v8.model.names[label_idx] + f" {conf:.2f}"
        cv2.putText(img, _text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, _color, 2)

    image = Image.fromarray(np.uint8(img))
    return image


def function_yolo_v5(image_file):
    image = Image.open(image_file)

    # Run the image through the model
    image = image.convert("RGB")
    image = np.array(image)
    results = model_v5(image)

    # directly use image
    # image = predictions.render()[0]
    # Get the predictions
    # pred = predictions

    # # Get the class labels and confidence scores
    pred = results.xyxy[0].detach().cpu().numpy()
    for obj in pred:
        x1, y1, x2, y2, conf, label_idx = obj
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        _color = color_list[int(label_idx) % len(color_list)]
        cv2.rectangle(image, pt1, pt2, _color, 2)
        _text = results.names[label_idx] + f" {conf:.2f}"
        cv2.putText(image, _text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, _color, 2)

    # Convert image from numpy array to PIL image
    image = Image.fromarray(np.uint8(image))
    return image


def main():

    st.set_page_config(page_title="Object Detection App", page_icon=":guardsman:", layout="wide")

    st.title("Object Detection App v5 vs v8")

    # Create a file uploader widget
    image_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    if image_file is not None:
        # Open the image file
        with col1:
            st.header(f"YOLOV5 - {CONST_MODEL_01}")
            image = function_yolo_v5(image_file)
            st.image(image, caption='V5 Result.', use_column_width=True)
        with col2:
            st.header(f"YOLOV8 - {CONST_MODEL_02}")
            image = function_yolo_v8(image_file)
            st.image(image, caption='V8 Result.', use_column_width=True)


if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
