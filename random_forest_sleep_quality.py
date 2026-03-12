import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
        description="Train a random forest model for the sleep ML project."
    )
    parser.add_argument(
        "--data",
        default="ML_proj_data.csv",
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument(
        "--target",
        default="sleep_quality",
        choices=["sleep_quality", "sleep_duration_hours"],
        help="Column to predict. Default is sleep_quality.",
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

    # plot the original target data before training
    plt.figure(figsize=(10, 5))
    plt.plot(df[args.target].values, marker="o", linestyle="-")
    plt.title(f"Original {args.target} Data")
    plt.xlabel("Sample Index")
    plt.ylabel(args.target)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # use every column except the target as input features
    X = df[['mood_rating', 'stress_level', 'mental_health_score','daily_screen_time_hours']]
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = build_model(n_estimators=args.trees, random_state=args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Mean Squared Error
    mse = mean_squared_error(y_test, preds)

    # Root Mean Squared Error
    rmse = mse ** 0.5

    # Mean Absolute Error
    mae = mean_absolute_error(y_test, preds)

    # R-squared
    r2 = r2_score(y_test, preds)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("=" * 60)
    print(f"Random Forest Results for target: {args.target}")
    print("=" * 60)
    print(f"Rows in dataset: {len(df)}")
    print(f"Training rows:   {len(X_train)}")
    print(f"Testing rows:    {len(X_test)}")
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
    print("Top feature importances")
    for _, row in importance_df.iterrows():
        print(f"- {row['feature']}: {row['importance']:.4f}")

    out_file = data_path.parent / f"{args.target}_Results.csv"
    importance_df.to_csv(out_file, index=False)
    print()
    print(f"Saved feature importances to: {out_file}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", marker="o")
    plt.plot(preds, label="Predicted", marker="x")
    plt.title(f"Actual vs Predicted: {args.target}")
    plt.xlabel("Test Sample Index")
    plt.ylabel(args.target)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
