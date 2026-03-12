import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv") #put in the actual correct data set

#TODO: define target from  the data set
#TODO: define predictors from the data set
#TODO: we can run a correlation matrix

def multiple(target, predictors):
    X = predictors
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2:", r2_score(y_test, y_pred))


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
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
        'Predictor': predictor,
        'R2 Score': r2,
        'MSE' : mse,
        'RMSE': rmse,
        'Intercept': model.intercept_,
        'Slope': model.coef_[0]
        })
    

    results_df = pd.DataFrame(results)
    print("\nSummary of all predictors:")
    print(results_df)


