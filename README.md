# OncoPredict — Breast Tumor Classifier (Benign vs Malignant)

OncoPredict is a small machine learning project that predicts whether a breast tumor is benign or malignant. It includes:
- A training pipeline to train and persist a supervised classifier
- A Streamlit web app that accepts user input and returns a prediction
- Model explainability using SHAP:
  - Global feature importance (bar plot)
  - Per-patient explanation (SHAP waterfall plot)

This repository is intended for research / educational use and demonstrates an end-to-end ML workflow: data → model → explainability → web UI.

---

## Demo
https://cnjfbpmk8kek7qkvitdh7v.streamlit.app/

https://github.com/user-attachments/assets/ea602696-2837-4898-8be0-fcaa042762c7


## Features
- Predicts tumor status: benign or malignant (binary classification)
- Model-agnostic training script (example uses RandomForest / XGBoost)
- Streamlit dashboard:
  - Input form for required features
  - Predicted class and probability
  - Global feature importance (SHAP bar plot)
  - Per-patient SHAP waterfall plot explaining the prediction

---

## Contributing

Contributions welcome. Suggested workflow:
1. Open an issue describing the change / bug / enhancement
2. Create a feature branch: `git checkout -b feat/your-change`
3. Implement tests and make changes
4. Open a pull request with a clear description

---
