# 八哥辨識器（Myna detector）

範例專案：使用遷移學習 (MobileNetV2) 訓練一個簡單的八哥辨識器，並以 Streamlit 建立一個可部署的網頁介面。

目錄結構（範例）
```
Hw04/
  ├─ app.py
  ├─ train.py
  ├─ requirements.txt
  ├─ model/  # 儲存訓練好的 bird_model.h5
  └─ data/
      └─ train/
          ├─ bngo/      # 八哥影像 (class name 可自訂)
          └─ other/     # 其他非八哥影像
```

快速上手

1. 準備資料集

- 將你的影像放在 `data/train/<class_name>/` 子資料夾中，每個類別為一個資料夾。
- 建議至少每一類各 100 張以上影像以取得合理效果；或使用更多資料與資料擴增。

2. 本地訓練

```
python train.py --data_dir data/train --epochs 10 --batch_size 32
```

訓練完會將最佳模型儲存到 `model/bird_model.h5`。

3. 本地測試 Streamlit 應用

```
streamlit run app.py
```

在網頁上上傳影像，按下「辨識」即可看到是否為八哥的機率。

4. 部署到 Streamlit Cloud

- 將此資料夾放到 GitHub repo。
- 在 `streamlit.io` (Streamlit Community Cloud) 上登入並新增一個 App，連結此 GitHub repo。
- 設定 `Main file` 為 `app.py`。平台會自動使用 `requirements.txt` 安裝相依套件。

注意事項

- TensorFlow 在不同作業系統與 Python 版本可能需要特別安裝方式，若在 Windows 上安裝失敗，請參考 TensorFlow 官方文件。
- 若模型太大或推論時間較長，可改用 TensorFlow Lite 或將模型上傳到雲端並於服務端做推論。

參考

- 原始 Notebook: https://github.com/yenlung/AI-Demo/blob/master/%E3%80%90Demo02%E3%80%91%E9%81%B7%E7%A7%BB%E5%BC%8F%E5%AD%B8%E7%BF%92%E5%81%9A%E5%85%AB%E5%93%A5%E8%BE%A8%E8%AD%98%E5%99%A8.ipynb
