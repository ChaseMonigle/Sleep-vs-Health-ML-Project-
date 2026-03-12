import argparse
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def build_model(n_estimators, random_state):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ML_proj_data.csv")
    parser.add_argument("--target", default="sleep_quality")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = pd.read_csv(args.data)

    target = args.target

    if target not in df.columns:
        raise ValueError(f"{target} not found in dataset")

    results = []

    features = [c for c in df.columns if c != target]

    for feature in features:

        X = df[[feature]]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed
        )

        model = build_model(args.trees, args.seed)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            "variable_tested": feature,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"Finished testing: {feature}")

    results_df = pd.DataFrame(results)

    output_file = "single_variable_rf_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\nAll results saved to:", output_file)


if __name__ == "__main__":
    main()
