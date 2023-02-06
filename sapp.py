"""
Main UI streamlit app
"""
import os
import copy
import sys
from streamlit.web import cli as stcli
from streamlit import runtime

import streamlit as st
from PIL import Image
import pandas as pd

from ObjectDetectorYoloV5 import ObjectDetectorYoloV5
from ObjectDetectorYoloV8 import ObjectDetectorYoloV8
from ObjectDetectorFromRestAPI import ObjectDetectorFromRestAPI
from compare_utility import *
from image_utility import get_image_compare_result


LIST_YOLOV5 = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
LIST_YOLOV8 = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

# LIST_YOLOV7 = ["yolov7-tiny", "yolov7"]
LIST_URL = []
OD_API_URL = os.environ.get('DIFFOD_URL', None)
lst_extra_od = []
if OD_API_URL is not None:
    url_item = OD_API_URL
    print(url_item)
    od_extra_api = ObjectDetectorFromRestAPI({'api_url': url_item})
    dic_models = od_extra_api.get_supported_models()
    lst_extra_od = [item['model_name'] for item in dic_models.get('supported_models', [])]
    pass

ALL_MODELS = LIST_YOLOV5 + lst_extra_od + LIST_YOLOV8

SET_YOLOV5 = set(LIST_YOLOV5)
SET_YOLOV8 = set(LIST_YOLOV8)
set_extra_od = set(lst_extra_od)

# perset the models for 1at launch
idx_model_a = ALL_MODELS.index("yolov5s")
idx_model_b = ALL_MODELS.index("yolov8s")


def get_od_model(model_name: str):
    global od_extra_api
    if model_name in SET_YOLOV5:
        return ObjectDetectorYoloV5({'model_name': model_name})
    if model_name in SET_YOLOV8:
        return ObjectDetectorYoloV8({'model_name': model_name})
    if model_name in set_extra_od:
        return ObjectDetectorFromRestAPI({'api_url': url_item, 'model_name': model_name})


# initialize A B models
model_a = get_od_model(ALL_MODELS[idx_model_a])
model_b = get_od_model(ALL_MODELS[idx_model_b])


def on_change_model():
    print(st.session_state['model_a'])
    print(st.session_state['model_b'])
    refresh_model(st.session_state['model_a'], st.session_state['model_b'])
    pass


def refresh_model(model_a_name, model_b_name):
    global model_a, model_b
    if model_a_name != model_a.model_name:
        print(f"RENEW MODEL A AS {model_a_name}")
        model_a.clear_model()
        model_a = get_od_model(model_a_name)

    if model_b_name != model_b.model_name:
        print(f"RENEW MODEL B AS {model_b_name}")
        model_b.clear_model()
        model_b = get_od_model(model_b_name)
    pass


def get_df_stat(lst_r_1, lst_r_2):
    """
    compare count of the classes
    """
    global model_a, model_b
    df1 = pd.DataFrame(lst_r_1)
    df2 = pd.DataFrame(lst_r_2)

    if model_a.model_name == model_b.model_name:
        df1['model'], df2['model'] = model_a.model_name + '_L', model_b.model_name + '_R'
    else:
        df1['model'], df2['model'] = model_a.model_name, model_b.model_name

    df = pd.concat([df1, df2])
    df_stat = df.groupby(['label', 'model']).agg({'conf': 'count'}).reset_index().pivot(columns='model', index='label', values='conf')
    df_stat = df_stat.fillna(0)
    for col in df_stat.columns.values:
        df_stat[col] = df_stat[col].astype('int')

    try:
        df_stat = df_stat = df_stat[[model_a.model_name, model_b.model_name]]
    except:
        pass
    return df_stat


def main():
    global model_b, model_a
    st.set_page_config(page_title="Diff Object Detection", page_icon="figure/compare.png", layout="wide")

    st.title("Diff Object Detection")

    # Create a file uploader widget
    # image_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    st.sidebar.subheader("Choose A/B Models:")
    result_model_a = st.sidebar.selectbox(
        "Model A", ALL_MODELS, index=idx_model_a, on_change=on_change_model, key='model_a')

    with st.sidebar.expander("Model A config"):
        _a_iou_input = st.slider(label='IOU threshold',
                                 max_value=1.0,
                                 min_value=0.0,
                                 value=0.45,
                                 step=0.05,
                                 key='model_a_iou')
        _a_conf_input = st.slider(label='Conf threshold',
                                        max_value=1.0,
                                        min_value=0.0,
                                        value=0.25,
                                        step=0.05,
                                        key='model_a_conf')
        result_model_b = st.sidebar.selectbox(
            "Model B", ALL_MODELS, index=idx_model_b, on_change=on_change_model, key='model_b')

    with st.sidebar.expander("Model B config"):

        _b_iou_input = st.slider(label='IOU threshold', max_value=1.0, min_value=0.0, value=0.45, step=0.05, key='model_b_iou')
        _b_conf_input = st.slider(label='Conf threshold', max_value=1.0, min_value=0.0, value=0.25, step=0.05, key='model_b_conf')

    image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    col1, col2, col3 = st.columns(3)
    if image_file is not None:
        # Open the image file
        _temp_name = 'temp.jpg'
        image = Image.open(image_file)
        image.save(_temp_name)
        refresh_model(st.session_state['model_a'], st.session_state['model_b'])

        dic_cfg_a = {"iou_thres": st.session_state['model_a_iou'], "conf_thres": st.session_state['model_a_conf']}
        dic_cfg_b = {"iou_thres": st.session_state['model_b_iou'], "conf_thres": st.session_state['model_b_conf']}

        lst_result_a = model_a.inference_as_json_by_filepath(_temp_name, dic_cfg=dic_cfg_a)

        lst_result_b = model_b.inference_as_json_by_filepath(_temp_name, dic_cfg=dic_cfg_b)
        df_stat = get_df_stat(lst_result_a, lst_result_b)

        with col1:
            st.subheader(f"Model A - {model_a.model_name}")
            image_1 = get_image_compare_result(_temp_name, copy.deepcopy(lst_result_a), copy.deepcopy(lst_result_b))
            st.image(image_1, caption=f'{model_a.model_name} Result.', use_column_width=True)
        with col2:
            st.subheader(f"Model B - {model_b.model_name}")
            image_2 = get_image_compare_result(_temp_name, lst_result_b, lst_result_a, color_diff=(0, 0, 255))
            st.image(image_2, caption=f'{model_b.model_name} Result.', use_column_width=True)
        with col3:
            st.subheader(f"Labels Count:")
            st.dataframe(df_stat.style.highlight_max(axis=1))


if __name__ == '__main__':
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
