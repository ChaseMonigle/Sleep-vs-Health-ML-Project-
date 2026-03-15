import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "Age",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
]

TARGET_COLUMN = "Quality of Sleep"


def build_model(n_estimators: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a random forest model to predict Quality of Sleep."
    )
    parser.add_argument(
        "--data",
        default="Sleep_health_and_lifestyle_dataset.csv",
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for testing.",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=300,
        help="Number of trees in the random forest.",
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

    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # keep only the variables you listed
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # simple cleanup in case there are blanks
    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = build_model(n_estimators=args.trees, random_state=args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("=" * 60)
    print(f"Random Forest Results for target: {TARGET_COLUMN}")
    print("=" * 60)
    print(f"Rows in dataset: {len(df)}")
    print(f"Training rows:   {len(X_train)}")
    print(f"Testing rows:    {len(X_test)}")
    print()
    print("Features used")
    for col in FEATURE_COLUMNS:
        print(f"- {col}")
    print()
    print("Model settings")
    print(f"- Trees:        {args.trees}")
    print(f"- Test size:    {args.test_size}")
    print(f"- Random seed:  {args.seed}")
    print()
    print("Evaluation metrics")
    print(f"- MAE:   {mae:.4f}")
    print(f"- MSE:   {mse:.4f}")
    print(f"- RMSE:  {rmse:.4f}")
    print(f"- R^2:   {r2:.4f}")
    print()
    print("Feature importances")
    for _, row in importance_df.iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")

    out_file = data_path.parent / "quality_of_sleep_feature_importances.csv"
    importance_df.to_csv(out_file, index=False)
    print()
    print(f"Saved feature importances to: {out_file}")


if __name__ == "__main__":
    main()
