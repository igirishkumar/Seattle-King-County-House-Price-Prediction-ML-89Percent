# modeling_all_models.py
# Trains 7 ML models on all 8 pre-processed versions
# Saves every model + best ones

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# === 1. CONFIG ===
DATA_DIR = "versions"
MODEL_DIR = "models_all"
os.makedirs(MODEL_DIR, exist_ok=True)

versions = [
    "Raw_All", "Raw_Selected", "Out_All", "Out_Selected",
    "Norm_All", "Norm_Selected", "Clean_All", "Clean_Selected"
]

# Model-specific version preference (to reduce overfitting)
MODEL_VERSION = {
    "Linear Regression": "Clean_All",        # Needs scaling + no high-dim
    "Decision Tree":     "Out_Selected",     # Trees love clean data
    "Random Forest":     "Clean_Selected",   # Best combo
    "Gradient Boosting": "Clean_Selected",
    "XGBoost":           "Clean_Selected",
    "CatBoost":         "Clean_All",
    "AdaBoost":          "Out_All"           # Simpler data
}

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=10, min_samples_leaf= 2, min_samples_split= 10, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, min_samples_leaf= 2, min_samples_split= 2),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=7, random_state=42, learning_rate= 0.05, subsample= 0.8),
    "XGBoost":           XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42, colsample_bytree = 0.8, subsample= 0.8),
    "CatBoost":         CatBoostRegressor(iterations=500, depth=8, learning_rate=0.1, random_state=42, l2_leaf_reg= 1),
    "AdaBoost":          AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100, learning_rate= 0.5, random_state=42)
}

# === 3. Train & Save All ===
results = {}
best_overall_mae = np.inf
best_model = None
best_name = None
best_version = None

for model_name, model in models.items():
    v = MODEL_VERSION[model_name]
    print(f"\nTraining {model_name} on {v}...")

    # Load data
    train_path = os.path.join(DATA_DIR, f"{v}_train.csv")
    test_path  = os.path.join(DATA_DIR, f"{v}_test.csv")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    y_train = np.log1p(train_df['price'])
    X_train = train_df.drop('price', axis=1)
    y_test  = np.log1p(test_df['price'])
    X_test  = test_df.drop('price', axis=1)

    # Fit
    model.fit(X_train, y_train)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    key = f"{model_name}__{v}"
    results[key] = {"MAE": mae, "R2": r2, "model": model, "version": v}

    # Save model
    model_path = os.path.join(MODEL_DIR, f"model_{model_name.replace(' ', '_')}__{v}.pkl")
    joblib.dump(model, model_path)

    print(f"   MAE: ${mae:,.0f} | R²: {r2:.4f} → saved: {os.path.basename(model_path)}")

    # Track best
    if mae < best_overall_mae:
        best_overall_mae = mae
        best_model = model
        best_name = model_name
        best_version = v
        best_path = model_path


results = []

print("Training 7 models and computing % metrics...")

for name, model in models.items():
    v = MODEL_VERSION[name]
    print(f"  → {name:<15} on {v}")

    # Load data
    train_path = os.path.join(DATA_DIR, f"{v}_train.csv")
    test_path  = os.path.join(DATA_DIR, f"{v}_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"     Missing data for {v}")
        continue

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Actual prices (original scale)
    actual_train = train_df['price']
    actual_test  = test_df['price']

    # Mean price for percentage
    mean_price_train = actual_train.mean()
    mean_price_test  = actual_test.mean()

    # Log transform
    y_train = np.log1p(train_df['price'])
    X_train = train_df.drop('price', axis=1)
    y_test  = np.log1p(test_df['price'])
    X_test  = test_df.drop('price', axis=1)

    # Fit
    model.fit(X_train, y_train)

    # Predictions
    pred_train = np.expm1(model.predict(X_train))
    pred_test  = np.expm1(model.predict(X_test))

    # Metrics function
    def calc_metrics(y_true, y_pred, mean_price):
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {
            "MAE":  mae,
            "MAE%": mae / mean_price * 100,
            "MSE":  mse,
            "MSE%": mse / (mean_price ** 2) * 100,
            "RMSE": rmse,
            "RMSE%": rmse / mean_price * 100,
            "R2":   r2_score(y_true, y_pred)
        }

    train_metrics = calc_metrics(actual_train, pred_train, mean_price_train)
    test_metrics  = calc_metrics(actual_test,  pred_test,  mean_price_test)

    # Append
    for split, metrics, mean_price in [
        ("Train", train_metrics, mean_price_train),
        ("Test",  test_metrics,  mean_price_test)
    ]:
        results.append({
            "Model": name,
            "Version": v,
            "Split": split,
            "MAE":  metrics["MAE"],
            "MAE%": metrics["MAE%"],
            "MSE":  metrics["MSE"],
            "MSE%": metrics["MSE%"],
            "RMSE": metrics["RMSE"],
            "RMSE%": metrics["RMSE%"],
            "R2":   metrics["R2"]
        })


# === FINAL TABLE ===
df_results = pd.DataFrame(results)

# Format
df_results["MAE"]   = df_results["MAE"].round(0).astype(int)
df_results["MAE%"]  = df_results["MAE%"].round(2)
df_results["MSE"]   = df_results["MSE"].round(0).astype(int)
df_results["MSE%"]  = df_results["MSE%"].round(2)
df_results["RMSE"]  = df_results["RMSE"].round(0).astype(int)
df_results["RMSE%"] = df_results["RMSE%"].round(2)
df_results["R2"]    = df_results["R2"].round(4)

# Sort
df_results = df_results.sort_values(["Model", "Version", "Split"]).reset_index(drop=True)

# === DISPLAY ===
print("\n" + "="*130)
print("FULL METRICS WITH % RELATIVE TO MEAN PRICE (7 MODELS)".center(130))
print("="*130)
print(df_results.to_string(index=False))

# === SAVE ===
df_results.to_csv("model_metrics_with_percent.csv", index=False)
print(f"\nResults saved → model_metrics_with_percent.csv")

