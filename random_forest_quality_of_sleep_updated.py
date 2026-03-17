import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "daily_screen_time_hours",
    "sleep_duration_hours",
    "mood_rating",
    "stress_level",
    "physical_activity_hours_per_week",
    "mental_health_score",
    "caffeine_intake_mg_per_day",
    "weekly_anxiety_score",
    "weekly_depression_score",
    "mindfulness_minutes_per_day",
]

TARGET_COLUMN = "sleep_quality"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # keep rows inside the valid scoring ranges
    df = df[
        df["sleep_quality"].between(1, 10) &
        df["mood_rating"].between(1, 10) &
        df["stress_level"].between(1, 10) &
        df["mental_health_score"].between(0, 100) &
        df["weekly_anxiety_score"].between(0, 20) &
        df["weekly_depression_score"].between(0, 20)
    ]

    # keep only non-negative values for these numeric fields
    df = df[
        (df["sleep_duration_hours"] >= 0) &
        (df["daily_screen_time_hours"] >= 0) &
        (df["physical_activity_hours_per_week"] >= 0) &
        (df["caffeine_intake_mg_per_day"] >= 0) &
        (df["mindfulness_minutes_per_day"] >= 0)
    ]

    # remove duplicate rows
    df = df.drop_duplicates()

    return df


def print_metrics(name: str, y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"R2:   {r2:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    return {
        "Model": name,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and tune a Random Forest model to predict sleep_quality."
    )
    parser.add_argument(
        "--data",
        default="ML_proj_data.csv",
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable results.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset: {data_path}")

    df = pd.read_csv(data_path)

    print("Original shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())

    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = clean_data(df)

    print("\nShape after cleaning:", df.shape)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # untuned model
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            random_state=args.seed,
            n_jobs=-1
        ))
    ])

    rf_pipe.fit(X_train, y_train)
    untuned_preds = rf_pipe.predict(X_test)
    untuned_results = print_metrics("Untuned Random Forest", y_test, untuned_preds)

    # 5-fold CV for untuned model
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    untuned_cv_r2 = cross_val_score(
        rf_pipe, X, y, cv=kfold, scoring="r2"
    )
    untuned_cv_rmse = np.sqrt(
        -cross_val_score(rf_pipe, X, y, cv=kfold, scoring="neg_mean_squared_error")
    )

    print("\n--- Untuned Random Forest 5-Fold Cross Validation ---")
    print("R2 scores:", np.round(untuned_cv_r2, 4))
    print("Mean CV R2:", round(untuned_cv_r2.mean(), 4))
    print("Std CV R2:", round(untuned_cv_r2.std(), 4))
    print("RMSE scores:", np.round(untuned_cv_rmse, 4))
    print("Mean CV RMSE:", round(untuned_cv_rmse.mean(), 4))
    print("Std CV RMSE:", round(untuned_cv_rmse.std(), 4))

    # tuned model
    tuned_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(
            random_state=args.seed,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    grid = GridSearchCV(
        tuned_pipe,
        param_grid=param_grid,
        cv=kfold,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    tuned_preds = best_rf.predict(X_test)
    tuned_results = print_metrics("Tuned Random Forest", y_test, tuned_preds)

    print("\nBest Random Forest Params:")
    print(grid.best_params_)

    tuned_cv_r2 = cross_val_score(
        best_rf, X, y, cv=kfold, scoring="r2"
    )
    tuned_cv_rmse = np.sqrt(
        -cross_val_score(best_rf, X, y, cv=kfold, scoring="neg_mean_squared_error")
    )

    print("\n--- Tuned Random Forest 5-Fold Cross Validation ---")
    print("R2 scores:", np.round(tuned_cv_r2, 4))
    print("Mean CV R2:", round(tuned_cv_r2.mean(), 4))
    print("Std CV R2:", round(tuned_cv_r2.std(), 4))
    print("RMSE scores:", np.round(tuned_cv_rmse, 4))
    print("Mean CV RMSE:", round(tuned_cv_rmse.mean(), 4))
    print("Std CV RMSE:", round(tuned_cv_rmse.std(), 4))

    # comparison table
    comparison_df = pd.DataFrame({
        "Model": ["Untuned Random Forest", "Tuned Random Forest"],
        "Test R2": [untuned_results["R2"], tuned_results["R2"]],
        "Test RMSE": [untuned_results["RMSE"], tuned_results["RMSE"]],
        "CV Mean R2": [untuned_cv_r2.mean(), tuned_cv_r2.mean()],
        "CV Mean RMSE": [untuned_cv_rmse.mean(), tuned_cv_rmse.mean()],
    })

    print("\nModel Comparison:")
    print(comparison_df)

    # feature importances from best model
    rf_model = best_rf.named_steps["model"]
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": rf_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df)

    out_file = data_path.parent / "sleep_quality_feature_importances.csv"
    importance_df.to_csv(out_file, index=False)

    compare_file = data_path.parent / "random_forest_model_comparison.csv"
    comparison_df.to_csv(compare_file, index=False)

    print(f"\nSaved feature importances to: {out_file}")
    print(f"Saved model comparison to: {compare_file}")

    # actual vs predicted plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, tuned_preds)
    plt.xlabel("Actual Sleep Quality")
    plt.ylabel("Predicted Sleep Quality")
    plt.title("Tuned Random Forest: Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # feature importance plot
    plt.figure(figsize=(10, 5))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()

    # CV comparison plot
    plt.figure(figsize=(6, 4))
    plt.bar(["Untuned RF", "Tuned RF"], [untuned_cv_r2.mean(), tuned_cv_r2.mean()])
    plt.ylabel("Mean CV R2")
    plt.title("Random Forest Cross-Validated R2 Comparison")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()