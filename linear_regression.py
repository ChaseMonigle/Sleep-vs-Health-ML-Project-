import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
df = pd.read_csv("ML_proj_data.csv")

print("Original shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------
# 2. CLEAN DATA / APPLY VALID RANGES
# ---------------------------------------------------
df = df[
    df["sleep_quality"].between(1, 10) &
    df["mood_rating"].between(1, 10) &
    df["stress_level"].between(1, 10) &
    df["mental_health_score"].between(0, 100) &
    df["weekly_anxiety_score"].between(0, 20) &
    df["weekly_depression_score"].between(0, 20)
]

df = df[
    (df["sleep_duration_hours"] >= 0) &
    (df["daily_screen_time_hours"] >= 0) &
    (df["physical_activity_hours_per_week"] >= 0) &
    (df["caffeine_intake_mg_per_day"] >= 0) &
    (df["mindfulness_minutes_per_day"] >= 0)
]

df = df.drop_duplicates()

print("\nShape after cleaning invalid values:", df.shape)

# ---------------------------------------------------
# 3. SELECT FEATURES + TARGET
# ---------------------------------------------------
target_col = "sleep_quality"

feature_cols = [
    "daily_screen_time_hours",
    "sleep_duration_hours",
    "mood_rating",
    "stress_level",
    "physical_activity_hours_per_week",
    "mental_health_score",
    "caffeine_intake_mg_per_day",
    "weekly_anxiety_score",
    "weekly_depression_score",
    "mindfulness_minutes_per_day"
]

X = df[feature_cols]
y = df[target_col]

# ---------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------------------------------
# 5. HELPER FUNCTION
# ---------------------------------------------------
def print_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} ---")
    print("R2:   ", round(r2, 4))
    print("MSE:  ", round(mse, 4))
    print("RMSE: ", round(rmse, 4))
    print("MAE:  ", round(mae, 4))

# ---------------------------------------------------
# 6. LINEAR REGRESSION
# ---------------------------------------------------
linear_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_pipe.fit(X_train, y_train)
linear_pred = linear_pipe.predict(X_test)

print_metrics("Linear Regression Test Results", y_test, linear_pred)

# ---------------------------------------------------
# 7. 5-FOLD CROSS VALIDATION
# ---------------------------------------------------
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

linear_cv_r2 = cross_val_score(
    linear_pipe, X, y, cv=kfold, scoring="r2"
)

linear_cv_rmse = np.sqrt(
    -cross_val_score(linear_pipe, X, y, cv=kfold, scoring="neg_mean_squared_error")
)

print("\n--- Linear Regression 5-Fold Cross Validation ---")
print("R2 scores:", np.round(linear_cv_r2, 4))
print("Mean CV R2:", round(linear_cv_r2.mean(), 4))
print("Std CV R2:", round(linear_cv_r2.std(), 4))
print("RMSE scores:", np.round(linear_cv_rmse, 4))
print("Mean CV RMSE:", round(linear_cv_rmse.mean(), 4))
print("Std CV RMSE:", round(linear_cv_rmse.std(), 4))

# ---------------------------------------------------
# 8. RIDGE REGRESSION + TUNING
# ---------------------------------------------------
ridge_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

ridge_params = {
    "model__alpha": [0.01, 0.1, 1, 10, 100]
}

ridge_grid = GridSearchCV(
    ridge_pipe,
    ridge_params,
    cv=kfold,
    scoring="r2"
)

ridge_grid.fit(X_train, y_train)

best_ridge = ridge_grid.best_estimator_
ridge_pred = best_ridge.predict(X_test)

print("\nBest Ridge alpha:", ridge_grid.best_params_)
print_metrics("Tuned Ridge Test Results", y_test, ridge_pred)

# ---------------------------------------------------
# 9. RIDGE CROSS VALIDATION
# ---------------------------------------------------
ridge_cv_r2 = cross_val_score(
    best_ridge, X, y, cv=kfold, scoring="r2"
)

ridge_cv_rmse = np.sqrt(
    -cross_val_score(best_ridge, X, y, cv=kfold, scoring="neg_mean_squared_error")
)

print("\n--- Ridge Regression 5-Fold Cross Validation ---")
print("R2 scores:", np.round(ridge_cv_r2, 4))
print("Mean CV R2:", round(ridge_cv_r2.mean(), 4))
print("Std CV R2:", round(ridge_cv_r2.std(), 4))
print("RMSE scores:", np.round(ridge_cv_rmse, 4))
print("Mean CV RMSE:", round(ridge_cv_rmse.mean(), 4))
print("Std CV RMSE:", round(ridge_cv_rmse.std(), 4))

# ---------------------------------------------------
# 10. MODEL COMPARISON
# ---------------------------------------------------
comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Tuned Ridge"],
    "Test R2": [
        r2_score(y_test, linear_pred),
        r2_score(y_test, ridge_pred)
    ],
    "Test RMSE": [
        np.sqrt(mean_squared_error(y_test, linear_pred)),
        np.sqrt(mean_squared_error(y_test, ridge_pred))
    ],
    "CV Mean R2": [
        linear_cv_r2.mean(),
        ridge_cv_r2.mean()
    ],
    "CV Mean RMSE": [
        linear_cv_rmse.mean(),
        ridge_cv_rmse.mean()
    ]
})

print("\nModel Comparison:")
print(comparison_df)

# ---------------------------------------------------
# 11. COEFFICIENTS
# ---------------------------------------------------
linear_model = linear_pipe.named_steps["model"]
ridge_model = best_ridge.named_steps["model"]

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Linear Coefficient": linear_model.coef_,
    "Ridge Coefficient": ridge_model.coef_
})

print("\nCoefficients:")
print(coef_df.sort_values(by="Ridge Coefficient", key=abs, ascending=False))

# ---------------------------------------------------
# 12. PLOTS
# ---------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(y_test, linear_pred)
plt.xlabel("Actual Sleep Quality")
plt.ylabel("Predicted Sleep Quality")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_test, ridge_pred)
plt.xlabel("Actual Sleep Quality")
plt.ylabel("Predicted Sleep Quality")
plt.title("Ridge Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(["Linear", "Ridge"], [linear_cv_r2.mean(), ridge_cv_r2.mean()])
plt.ylabel("Mean CV R2")
plt.title("Cross-Validated R2 Comparison")
plt.tight_layout()
plt.show()