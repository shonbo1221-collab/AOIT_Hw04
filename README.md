# 八哥辨識器

這是一個基於 ONNX 模型的八哥辨識器，使用 Streamlit 作為前端框架。

## CRISP-DM 分析

### 1. 業務理解（Business Understanding）
目標是建立一個能夠辨識八哥種類的應用程式，幫助使用者快速分類三種八哥：Crested Myna, Javan Myna, Common Myna。

### 2. 資料理解（Data Understanding）
使用來自公開資料集的八哥影像，進行資料探索與視覺化，確保資料涵蓋三種類別，並且影像品質適合模型訓練。

### 3. 資料準備（Data Preparation）
- 將影像資料進行標籤化。
- 使用資料增強技術（如旋轉、縮放）來擴充資料集。
- 將資料分為訓練集與測試集。

### 4. 建模（Modeling）
- 使用遷移學習技術，選擇 MobileNetV2 作為基礎模型。
- 將模型轉換為 ONNX 格式，提升跨平台推論能力。

### 5. 評估（Evaluation）
- 使用測試集評估模型準確率。
- 可視化混淆矩陣，檢查分類錯誤。

### 6. 部署（Deployment）
- 使用 Streamlit 建立互動式應用程式。
- 部署至 Streamlit Cloud，提供即時推論服務。

## 功能
- 上傳圖片進行八哥種類辨識。
- 支援三種八哥分類：Crested Myna, Javan Myna, Common Myna。

## 部署
此應用程式已部署於 Streamlit Cloud，並可透過以下網址存取：

[Demo 網址](https://aoithw04-k7hfmbh3ojokk4y8oujpmg.streamlit.app/)

## 參考來源
- [GitHub: yenlung](https://github.com/yenlung)