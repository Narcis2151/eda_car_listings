import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

# Load the data
data_df_encoded = pd.read_csv("./data/processed_data.csv")

# Split the data
X = data_df_encoded.drop(columns=["price", "Log Price"])  # Features
y = data_df_encoded["Log Price"]  # Target (log-transformed price)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Feature Selection

# Correlation-Based Filtering
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
high_correlation_features = [
    column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)
]
X_selected = X.drop(columns=high_correlation_features)


X_train_selected = X_train[X_selected.columns]

# Univariate Feature Importance
# Use RandomForest to calculate feature importances
importance_model = RandomForestRegressor(random_state=42)
importance_model.fit(X_train, y_train)

# Rank features by importance
feature_importances = pd.Series(
    importance_model.feature_importances_, index=X_train.columns
)
top_features = feature_importances.nlargest(25).index

# Ensure top features exist in X_train_selected
top_features = [
    feature for feature in top_features if feature in X_train_selected.columns
]
X_train_selected = X_train_selected[top_features]

#Recursive Feature Elimination (RFE)
rfe_model = RandomForestRegressor(random_state=42)
rfe = RFE(estimator=rfe_model, n_features_to_select=20, step=1)

# Fit RFE on the subset of X_train_selected and y_train
rfe.fit(X_train_selected, y_train)

# Select final features based on RFE
selected_features_rfe = X_train_selected.columns[rfe.support_]
X_train_final = X_train_selected[selected_features_rfe]

# Apply the same transformation to X_test
X_test_final = X_test[X_train_final.columns]

print("Selected Features After RFE:", list(selected_features_rfe))

# Train Models (Without Hyperparameter Tuning) with Selected Features to get a sense of performance
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        random_state=42, max_depth=10, n_estimators=100
    ),
    "SVR": SVR(kernel="rbf"),
    "CatBoost": CatBoostRegressor(
        verbose=0, random_state=42, learning_rate=0.05, iterations=500, depth=4
    ),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=100),
    "LightGBM": LGBMRegressor(random_state=42, n_estimators=100),
    "MLP": MLPRegressor(random_state=42, max_iter=1000)
}

baseline_results = {}
for name, model in models.items():
    model.fit(X_train_final, y_train)

    predictions_test = model.predict(X_test_final)
    predictions_train = model.predict(X_train_final)

    rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))
    r2_test = r2_score(y_test, predictions_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
    r2_train = r2_score(y_train, predictions_train)

    baseline_results[name] = {
        "RMSE Test": rmse_test,
        "R² Test": r2_test,
        "RMSE Train": rmse_train,
        "R² Train": r2_train,
    }

print("Baseline Model Results:")
for name, metrics in baseline_results.items():
    print(
        f"{name}: RMSE Train = {metrics['RMSE Train']:.4f}, R² Train = {metrics['R² Train']:.4f}, "
        f"RMSE Test = {metrics['RMSE Test']:.4f}, R² Test = {metrics['R² Test']:.4f}"
    )
    
# Plot the baseline results
model_names = list(baseline_results.keys())
rmse_values = [baseline_results[name]['RMSE Test'] for name in model_names]
r2_values = [baseline_results[name]['R² Test'] for name in model_names]

# RMSE
plt.figure(figsize=(12, 6))
plt.bar(model_names, rmse_values, color='skyblue')
plt.title("Model Comparison: Test RMSE (Selected Features)")
plt.ylabel("RMSE")
plt.show()

# R²
plt.figure(figsize=(12, 6))
plt.bar(model_names, r2_values, color='lightgreen')
plt.title("Model Comparison: Test R² (Selected Features)")
plt.ylabel("R²")
plt.show()

# Re-train Models with Hyperparameter Tuning

# Define hyperparameter grids for tuning
param_grids = {
    "Linear Regression": {},
    "SVR": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
    "CatBoost": {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05],
        "depth": [4, 6],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05]
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05]
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01]
    }
}

tuned_results = {}
for name, model in models.items():
    print(f"Tuning hyperparameters for {name} with selected features...")
    grid_search = GridSearchCV(
        model, param_grids[name], scoring="neg_mean_squared_error", cv=3
    )
    grid_search.fit(X_train_final, y_train)
    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test_final)
    predictions_train = best_model.predict(X_train_final)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
    r2_train = r2_score(y_train, predictions_train)

    tuned_results[name] = {
        "Best Params": grid_search.best_params_,
        "RMSE Test": rmse,
        "R² Test": r2,
        "RMSE Train": rmse_train,
        "R² Train": r2_train,
    }
    
# Step 5: Compare Models
print("\nComparison of Hyperparameter Tuning:")
for name in models.keys():
    baseline_rmse = baseline_results[name]["RMSE Test"]
    tuned_rmse = tuned_results[name]["RMSE Test"]
    baseline_r2 = baseline_results[name]["R² Test"]
    tuned_r2 = tuned_results[name]["R² Test"]
    baseline_rmse_train = baseline_results[name]["RMSE Train"]
    tuned_rmse_train = tuned_results[name]["RMSE Train"]
    baseline_r2_train = baseline_results[name]["R² Train"]
    tuned_r2_train = tuned_results[name]["R² Train"]

    print(
        f"{name}: Untuned RMSE Train = {baseline_rmse_train:.4f}, Tuned RMSE Train = {tuned_rmse_train:.4f}"
    )
    print(
        f"{name}: Untuned R² Train = {baseline_r2_train:.4f}, Tuned R² Train = {tuned_r2_train:.4f}"
    )
    print(
        f"{name}: Untuned RMSE Test = {baseline_rmse:.4f}, Tuned RMSE Test = {tuned_rmse:.4f}"
    )
    print(
        f"{name}: Untuned R² Test = {baseline_r2:.4f}, Tuned R² Test = {tuned_r2:.4f}\n"
    )
    
# Get the table of results
results_df = pd.DataFrame(tuned_results).T
results_df = results_df[
    ["RMSE Train", "R² Train", "RMSE Test", "R² Test", "Best Params"]
]
print(results_df)

# Save the results to a CSV file
results_df.to_csv("./data/model_comparison_results.csv")

# RMSE
model_names = list(tuned_results.keys())
rmse_values = [tuned_results[name]["RMSE Test"] for name in model_names]
r2_values = [tuned_results[name]["R² Test"] for name in model_names]

plt.figure(figsize=(12, 6))
plt.bar(model_names, rmse_values, color="skyblue")
plt.title("Model Comparison: Test RMSE Tuned")
plt.ylabel("RMSE")
plt.show()

# R²
plt.figure(figsize=(12, 6))
plt.bar(model_names, r2_values, color="lightgreen")
plt.title("Model Comparison: Test R² Tuned")
plt.ylabel("R²")
plt.show()