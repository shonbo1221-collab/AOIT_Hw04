# 八哥辨識器 Streamlit Demo

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# 設定標題
st.title("八哥辨識器 ONNX Demo")

# 上傳影像
uploaded_file = st.file_uploader("請上傳一張圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 顯示上傳的影像
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的圖片", use_column_width=True)

    # 將影像轉換為模型輸入格式
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # 載入 ONNX 模型
    onnx_model_path = "model/myna_classifier.onnx"
    session = ort.InferenceSession(onnx_model_path)

    # 檢查模型的輸入形狀
    input_shape = session.get_inputs()[0].shape
    st.write(f"模型輸入需求形狀: {input_shape}")

    # 調整影像資料形狀以符合模型需求
    if len(input_shape) == 3:  # 如果模型不需要 batch_size 維度
        image_array = np.squeeze(image_array, axis=0)
    elif len(input_shape) == 4 and input_shape[0] is None:  # 如果模型需要 batch_size 維度
        pass  # 保持現有形狀
    else:
        st.error("輸入形狀與模型需求不匹配，請檢查模型或輸入資料！")

    # 推論
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    prediction = np.argmax(outputs[0])

    # 顯示結果
    class_names = ["Crested Myna", "Javan Myna", "Common Myna"]
    st.write(f"預測結果: {class_names[prediction]}")