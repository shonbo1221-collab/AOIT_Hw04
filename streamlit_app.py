# 八哥辨識器 Streamlit Demo

import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

# 設定標題
st.title("八哥辨識器 ONNX Demo")

# 上傳影像
uploaded_file = st.file_uploader("請上傳一張圖片", type=["jpg", "jpeg", "png"])

# 初始化長條圖的數據
chart_data = pd.DataFrame({
    'Class': ["Crested Myna", "Javan Myna", "Common Myna"],
    'Score': [0.0, 0.0, 0.0]
})

# 顯示初始長條圖
chart_placeholder = st.empty()
chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X('Class', sort=None),
    y=alt.Y('Score', scale=alt.Scale(domain=[0, 1]))
).properties(
    width=600,
    height=400
)
chart_placeholder.altair_chart(chart, use_container_width=True)

if uploaded_file is not None:
    # 顯示上傳的影像
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的圖片", use_column_width=True)

    # 將影像轉換為 RGB 格式，確保只有 3 個通道
    image = image.convert('RGB')

    # 將影像轉換為模型輸入格式
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # 添加 batch 維度

    # 載入 ONNX 模型
    onnx_model_path = "model/myna_classifier.onnx"
    session = ort.InferenceSession(onnx_model_path)

    # 推論
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    scores = outputs[0].flatten()

    # 更新長條圖數據
    chart_data['Score'] = scores
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Class', sort=None),
        y=alt.Y('Score', scale=alt.Scale(domain=[0, 1]))
    ).properties(
        width=600,
        height=400
    )
    chart_placeholder.altair_chart(chart, use_container_width=True)

    # 顯示每個類別的分數
    class_names = ["Crested Myna", "Javan Myna", "Common Myna"]
    for i, score in enumerate(scores):
        st.write(f"{class_names[i]}: {score:.4f}")

    # 顯示最高分數的標籤
    prediction = np.argmax(scores)
    st.write(f"預測結果: {chart_data['Class'][prediction]} (分數: {scores[prediction]:.4f})")