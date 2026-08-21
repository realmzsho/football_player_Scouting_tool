# ⚽ AI Football Scout & Position Classifier NTI project

🚀 **Live Demo:** [Click Here to Try the Web App](https://footballplayerscoutingtoolv100.streamlit.app/)

## 📖 Project Overview
This project is an integrated Machine Learning web application designed to act as a virtual football scout. It combines both **Supervised** and **Unsupervised Learning** algorithms to evaluate detailed player attributes, predict their optimal playing positions, and discover similar, cost-effective alternatives in the transfer market. 

## ✨ Core Features
* **Position Classifier (Supervised Learning):** Utilizes an **XGBoost** model trained on 44 distinct player attributes (including pace, shooting, defending, and mentality) to accurately predict a player's optimal role on the pitch.
* **Player Scout (Unsupervised Learning):** Employs **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Nearest Neighbors (KNN)** to find players with similar playing styles. It acts as a smart recommendation engine for finding younger or cheaper alternatives based on custom budget and age constraints.
* **Interactive Dashboard:** Built entirely with **Streamlit**, featuring a clean, responsive, multi-tab layout for intuitive data entry and seamless user experience.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Manipulation & Math:** Pandas, NumPy
* **Model Persistence:** Joblib

## 📂 Repository Structure
* `app.py`: The main Streamlit application script containing the full ML pipeline.
* `FC26_20250921.csv`: The core dataset serving the Unsupervised Player Scout engine.
* `xgb_model.pkl`: The trained Standard XGBoost classifier.
* `pca_final.pkl` & `scaler.pkl`: The pre-trained dimensionality reduction and scaling objects.
* `label_encoder.pkl`: Decodes the model's numerical output into readable position names.
* `requirements.txt`: The project dependencies required for cloud deployment.

## 👥 Prepared By
This project was developed collaboratively by:

1. **[Yousif Mustafa Saeed Abd El Wahab]** — GitHub: [@Zeradex](https://github.com/Zeradex)
2. **[Asser Samir Ahmed El-sisi]** — GitHub: [@aserelsisi1](https://github.com/aserelsisi1)
3. **[Jana Ahmed Mohammed Abd-Elghani]** — GitHub: [@janjw14-dev](https://github.com/janjw14-dev)
4. **Mohammed Osama Mohammed-Khalil Shehabeldin** — GitHub: [@realmzsho](https://github.com/realmzsho)
5. **[Ali Mohamed Ali Eladgham]** — GitHub: [@46alimohamed46-byte](https://github.com/46alimohamed46-byte)
