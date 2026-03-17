import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("ML_proj_data.csv")
target = df[["sleep_quality"]]

predictors = df[["daily_screen_time_hours", "sleep_duration_hours", "mood_rating", "stress_level", "physical_activity_hours_per_week", 
"mental_health_score", "caffeine_intake_mg_per_day", "weekly_anxiety_score", "mindfulness_minutes_per_day", "weekly_depression_score"]]
predictors1 = df[["stress_level", "mood_rating", "weekly_anxiety_score", "weekly_depression_score", "sleep_duration_hours"]]


# Ordinary Least Squares Implementation

class OrdinaryLeastSquares:

    def fit(self, X, y):

        # add column of ones for intercept
        X = np.hstack((np.ones((X.shape[0],1)), X))

        # compute theta using normal equation
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y


    def predict(self, X):

        X = np.hstack((np.ones((X.shape[0],1)), X))

        return X @ self.theta



def multiple(target, predictors):
    X = predictors.copy()
    y = target.values

    #correlation check
    corr = X.corr()
    print("\nCorrelation Matrix:\n", corr)

    # OPTIONAL: drop highly correlated features
   
   #convert to numpy
    X = X.values

    #standardize the data 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #add non-linear feautures
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = poly.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    print(len(y_test))
    print(len(X_test))
    model = OrdinaryLeastSquares()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    print("R2:", r2_score(y_test, y_pred))
    print("Predictions summary:")
    print("Min:", np.min(y_pred))
    print("Max:", np.max(y_pred))
    print("Unique values:", len(np.unique(np.round(y_pred, 3))))
   
    # predicted vs. actual
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.show()

    #residual plot
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()

    # residual distribution
    plt.figure()
    plt.hist(residuals, bins=20)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()

    




def one_at_a_time(target, predictors):

    results = []

    for predictor in predictors.columns:

        print(f"\n--- Testing predictor: {predictor} ---")

        X = predictors[[predictor]].values
        y = target.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = OrdinaryLeastSquares()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.append({
            "Predictor": predictor,
            "R2 Score": r2,
            "MSE": mse,
            "RMSE": rmse,
            "Intercept": model.theta[0],
            "Slope": model.theta[1]
        })

    results_df = pd.DataFrame(results)

    print("\nSummary of all predictors:")
    print(results_df)



multiple(target, predictors)

