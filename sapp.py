import copy
import sys
from streamlit.web import cli as stcli
from streamlit import runtime

import streamlit as st
from PIL import Image
import pandas as pd

from ObjectDetectorYoloV5 import ObjectDetectorYoloV5
from ObjectDetectorYoloV7 import ObjectDetectorYoloV7
from ObjectDetectorYoloV8 import ObjectDetectorYoloV8
from compare_utility import *
from image_utility import get_image_compare_result


LIST_YOLOV5 = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]
LIST_YOLOV8 = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

LIST_YOLOV7 = ["yolov7-tiny", "yolov7"]
ALL_MODELS = LIST_YOLOV5 + LIST_YOLOV7 + LIST_YOLOV8

SET_YOLOV5 = set(LIST_YOLOV5)
SET_YOLOV8 = set(LIST_YOLOV8)
SET_YOLOV7 = set(LIST_YOLOV7)

# perset the models for 1at launch 
idx_model_a = ALL_MODELS.index("yolov5s")
#idx_model_a = ALL_MODELS.index("yolov7-tiny")
idx_model_b = ALL_MODELS.index("yolov8s")


def get_od_model(model_name: str):
    if model_name in SET_YOLOV5:
        return ObjectDetectorYoloV5({'model_name': model_name})
    if model_name in SET_YOLOV8:
        return ObjectDetectorYoloV8({'model_name': model_name})
    if model_name in SET_YOLOV7:
        return ObjectDetectorYoloV7({'model_name': model_name})

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
    df1['model'] = model_a.model_name
    df2['model'] = model_b.model_name
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
    st.sidebar.text("Choose A/B Models:")
    result_v5 = st.sidebar.selectbox(
        "Model A", ALL_MODELS, index=idx_model_a, on_change=on_change_model, key='model_a')

    result_v8 = st.sidebar.selectbox(
        "Model B", ALL_MODELS, index=idx_model_b, on_change=on_change_model, key='model_b')

    image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    col1, col2, col3 = st.columns(3)
    if image_file is not None:
        # Open the image file
        _temp_name = 'temp.jpg'
        image = Image.open(image_file)
        image.save(_temp_name)
        refresh_model(st.session_state['model_a'], st.session_state['model_b'])

        lst_result_a = model_a.inference_as_json_by_filepath(_temp_name)
        lst_result_b = model_b.inference_as_json_by_filepath(_temp_name)
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
