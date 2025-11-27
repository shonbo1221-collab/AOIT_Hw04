import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "./model/bird_model.h5"
INPUT_SIZE = (224, 224)


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    model = tf.keras.models.load_model(path)
    return model


def preprocess_image(img: Image.Image, size=INPUT_SIZE):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, img: Image.Image):
    x = preprocess_image(img)
    preds = model.predict(x)
    if preds.shape[-1] == 1:
        prob = float(tf.nn.sigmoid(preds[0, 0]))
        return {"八哥": prob, "非八哥": 1 - prob}
    else:
        probs = tf.nn.softmax(preds[0]).numpy()
        # assume classes alphabetical or saved order
        return {"八哥": float(probs[0]), "非八哥": float(probs[1])}


def main():
    st.title("八哥辨識器 / Myna Detector")
    st.write("上傳一張鳥類照片，模型會預測是否為八哥。")

    model = load_model(MODEL_PATH)
    if model is None:
        st.warning("找不到模型權重 (model/bird_model.h5)。請先用 `train.py` 訓練或上傳模型到 `model/` 資料夾。")

    uploaded = st.file_uploader("上傳影像", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("尚未上傳影像。可在 README 中查看如何準備資料並訓練模型。")
        return

    image = Image.open(uploaded)
    st.image(image, caption="上傳影像", use_column_width=True)

    if st.button("辨識"):
        if model is None:
            st.error("目前沒有可用的模型。請先訓練模型或上傳 `model/bird_model.h5`。")
            return
        with st.spinner("正在辨識..."):
            results = predict(model, image)
        st.subheader("預測結果")
        for k, v in results.items():
            st.write(f"- {k}: {v*100:.2f}%")


if __name__ == "__main__":
    main()
