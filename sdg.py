# 🌍 SDG 13: Climate Action – Predicting Greenhouse Gas Emissions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv("D:\Downloads\FAO_EMSTOT.csv")

# Step 2: Select relevant features and drop missing values
df = df[['REF_AREA_LABEL', 'URBANISATION_LABEL', 'COMP_BREAKDOWN_1_LABEL',
         'COMP_BREAKDOWN_2_LABEL', 'TIME_PERIOD', 'OBS_VALUE']].dropna()

# Step 3: Feature and target separation
X = df.drop("OBS_VALUE", axis=1)
y = df["OBS_VALUE"]

# Step 4: Preprocess categorical features using OneHotEncoding
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Step 5: Create a Random Forest regression pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
pipeline.fit(X_train, y_train)

# Step 8: Predict on test set
y_pred = pipeline.predict(X_test)

# Step 9: Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Evaluation Metrics")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Step 10: Visualize predictions
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual GHG Emissions")
plt.ylabel("Predicted GHG Emissions")
plt.title("Actual vs Predicted GHG Emissions")
plt.grid(True)
plt.tight_layout()
plt.show()
