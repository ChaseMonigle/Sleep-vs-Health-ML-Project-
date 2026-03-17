<<<<<<< HEAD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
target = df[["Quality of Sleep"]]

predictors = df[["Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"]]

corr_matrix = df.corr(numeric_only=True)

# print in terminal (optional)
print(corr_matrix)


def multiple(target, predictors):
    X = predictors
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2:", r2_score(y_test, y_pred))
    # Create an index for the test samples
    indices = np.arange(len(y_test))

    plt.figure(figsize=(10,6))

    # Plot actual values
    plt.scatter(indices, y_test, color='blue', label='Actual', alpha=0.7)

    # Plot predicted values
    plt.scatter(indices, y_pred, color='red', label='Predicted', alpha=0.7)

    plt.xlabel('Test Sample Index')
    plt.ylabel('Target Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()



def one_at_a_time(predictors, target):
    results = []
    for predictor in predictors:
        print(f"\n--- Testing predictor: {predictor} ---")
    
        # Prepare X and y
        X = df[[predictor]]  # single predictor at a time
        y = target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split data

        model = LinearRegression() #create model
        model.fit(X_train, y_train) #train model
        y_pred = model.predict(X_test) #predictions

        #performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.append({
        'Predictor': predictor,
        'R2 Score': r2,
        'MSE' : mse,
        'RMSE': rmse,
        'Intercept': model.intercept_,
        'Slope': model.coef_[0]
        })
        indices = np.arange(len(y_test))

        plt.figure(figsize=(10,6))

        # Plot actual values
        plt.scatter(indices, y_test, color='blue', label='Actual', alpha=0.7)

        # Plot predicted values
        plt.scatter(indices, y_pred, color='red', label='Predicted', alpha=0.7)

        plt.xlabel('Test Sample Index')
        plt.ylabel('Target Value')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.show()
    

    results_df = pd.DataFrame(results)
    print("\nSummary of all predictors:")
    print(results_df)
=======
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
print("\nMissing values:")
print(df.isnull().sum())

# ---------------------------------------------------
# 2. CLEAN DATA / APPLY VALID RANGES
# ---------------------------------------------------
# Keep only rows inside the expected scoring ranges
# You can change this if your professor wants clipping instead of dropping bad rows

df = df[
    df["sleep_quality"].between(1, 10) &
    df["mood_rating"].between(1, 10) &
    df["stress_level"].between(1, 10) &
    df["mental_health_score"].between(0, 100) &
    df["weekly_anxiety_score"].between(0, 20) &
    df["weekly_depression_score"].between(0, 20)
]

# Optional: if you want to make sure hours and caffeine are not negative
df = df[
    (df["sleep_duration_hours"] >= 0) &
    (df["daily_screen_time_hours"] >= 0) &
    (df["physical_activity_hours_per_week"] >= 0) &
    (df["caffeine_intake_mg_per_day"] >= 0) &
    (df["mindfulness_minutes_per_day"] >= 0)
]

print("\nShape after cleaning invalid values:", df.shape)

# Drop duplicates
df = df.drop_duplicates()

# ---------------------------------------------------
# 3. ENCODE CATEGORICAL YES/NO COLUMNS
# ---------------------------------------------------
# Adjust these mappings if your CSV uses different spelling/cases

yes_no_map = {
    "yes": 1, "no": 0,
    "Yes": 1, "No": 0,
    "Y": 1, "N": 0,
    True: 1, False: 0
}

df["uses_wellness_apps"] = df["uses_wellness_apps"].map(yes_no_map)
df["eats_healthy"] = df["eats_healthy"].map(yes_no_map)

print("\nUnique values after encoding:")
print("uses_wellness_apps:", df["uses_wellness_apps"].dropna().unique())
print("eats_healthy:", df["eats_healthy"].dropna().unique())

# ---------------------------------------------------
# 4. SELECT FEATURES + TARGET
# ---------------------------------------------------
target_col = "sleep_quality"

feature_cols = [
    "daily_screen_time_hours",
    "sleep_duration_hours",
    "mood_rating",
    "stress_level",
    "physical_activity_hours_per_week",
    "mental_health_score",
    "uses_wellness_apps",
    "eats_healthy",
    "caffeine_intake_mg_per_day",
    "weekly_anxiety_score",
    "weekly_depression_score",
    "mindfulness_minutes_per_day"
]

X = df[feature_cols]
y = df[target_col]

# ---------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------------------------------
# 6. HELPER FUNCTION
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
# 7. LINEAR REGRESSION PIPELINE
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
# 8. K-FOLD CROSS VALIDATION FOR LINEAR REGRESSION
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
# 9. RIDGE REGRESSION PIPELINE
# ---------------------------------------------------
ridge_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

# Tune alpha
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
# 10. K-FOLD CROSS VALIDATION FOR BEST RIDGE
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
# 11. COMPARE LINEAR VS RIDGE
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
# 12. COEFFICIENTS
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
# 13. PLOTS
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

# Bar chart for CV comparison
plt.figure(figsize=(6, 4))
plt.bar(["Linear", "Ridge"], [linear_cv_r2.mean(), ridge_cv_r2.mean()])
plt.ylabel("Mean CV R2")
plt.title("Cross-Validated R2 Comparison")
plt.tight_layout()
plt.show()
>>>>>>> 308d6da (update)
