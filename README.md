# 🏠 Household Wealth Index Predictor (DHS Dataset)

A production-ready machine learning web application built with **XGBoost**, **Scikit-Learn**, and **Streamlit** to predict household wealth indices (`hv270`) using Demographic and Health Survey (DHS) data indicators.

---

## 🚀 Live Demo & Overview
This application takes key household survey characteristics (such as drinking water source, toilet type, electricity, and major asset ownership) and instantly classifies the household into one of five wealth tiers (from Poorest to Richest).

* **Model Algorithm:** XGBoost Classifier
* **Hyperparameter Tuning:** Bayesian Optimization via Optuna
* **Frontend Framework:** Streamlit
* **Performance:** ~66% Test Accuracy across 5 distinct wealth tiers

---

## 📂 Project Structure
```text
├── app.py                             # Streamlit web application script
├── best_wealth_predictor_pipeline.pkl  # Trained Scikit-Learn / XGBoost pipeline
└── requirements.txt                   # Python package dependencies
