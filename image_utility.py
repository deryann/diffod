from compare_utility import *
import math
import cv2
import numpy as np
from PIL import Image


FONT_SCALE = 1.5 * 1e-3  # Adjust for larger font size in all images
FONT_THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
LINE_THICKNESS_SCALE = 1e-2  # Adjust for larger thickness in all images
TEXT_Y_OFFSET_SCALE = 1e-2  # Adjust for larger Y-offset of text and bounding box


def draw_bbox_by_dict_obj(img, obj, color):
    x1, y1, x2, y2, conf, label = obj['x1'], obj['y1'], obj['x2'], obj['y2'], obj['conf'], obj['label']
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    height, width, _ = img.shape
    font_thickness = math.ceil(min(width, height) * FONT_THICKNESS_SCALE)
    line_thickness = math.ceil(min(width, height) * LINE_THICKNESS_SCALE)
    cv2.rectangle(img, pt1, pt2, color, line_thickness)
    _text = label + f" {conf:.2f}"
    cv2.putText(img, _text, (pt1[0], pt1[1] - int(height * TEXT_Y_OFFSET_SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX,
                min(width, height) * FONT_SCALE,
                color, font_thickness)
    return img


def get_image_compare_result(filename: str, lst_base, lst_to, color_same=(0, 255, 0), color_diff=(255, 0, 0)):

    image = Image.open(filename)
    image = image.convert("RGB")
    image = np.array(image)

    lst_results = get_iou_compare_result(lst_base, lst_to)
    
    # Draw color_same first

    for obj in lst_results:
        if obj.get('link', None) is not None:
            image = draw_bbox_by_dict_obj(image, obj, color_same)

    for obj in lst_results:
        if obj.get('link', None) is None:
            image = draw_bbox_by_dict_obj(image, obj, color_diff)

    # Convert image from numpy array to PIL image
    image = Image.fromarray(np.uint8(image))
    return image
