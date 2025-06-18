# sdgMachineLearning
# 🌍 SDG 13 Climate Action – GHG Emissions Predictor

## 🔍 Project Overview
This machine learning project addresses **SDG 13: Climate Action** by predicting **greenhouse gas (GHG) emissions** using a dataset from the Food and Agriculture Organization (FAO). By leveraging a **Random Forest Regressor**, the model identifies historical patterns and helps policymakers forecast emissions based on urbanization, geography, and sector data.

## ✅ Problem Statement
Greenhouse gas emissions are a primary contributor to climate change. Governments, environmental organizations, and climate scientists need data-driven insights to mitigate their effects. This tool provides emissions estimates using historical data to support proactive climate action.

## 🧠 Machine Learning Approach
- **Model**: Random Forest Regressor (Supervised Learning)
- **Pipeline**: 
  - Data Preprocessing (categorical encoding)
  - Model training & prediction
  - Evaluation with R², MAE, RMSE
- **Tools**: Python, Pandas, Scikit-learn, Seaborn, Matplotlib

## 🖼️ Demo Screenshots
| Data Preview | Model Evaluation | Prediction Graph |
|--------------|------------------|------------------|
| ![Data](screenshots/data.png) | ![Eval](screenshots/metrics.png) | ![Graph](screenshots/plot.png) |

## 📁 Files Included
- `SDG13_GHG_Emissions_Predictor.ipynb`: Jupyter notebook with full code
- `model.py`: Standalone Python script version
- `README.md`: Project overview and instructions
- `screenshots/`: Folder with demo screenshots

## 🚀 How to Run
```bash
pip install -r requirements.txt
python model.py
