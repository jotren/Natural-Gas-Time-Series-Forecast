import mlflow
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

def evaluate_model(predictions, y_test, run=None):
    if run is None and mlflow.active_run() is None:
        run = mlflow.start_run()  # Start a new run if none is active
    elif run is None:
        run = mlflow.active_run()  # Use the existing active run

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    regression_model = LinearRegression(fit_intercept=False)
    regression_model.fit(predictions.reshape(-1, 1), y_test)
    gradient = regression_model.coef_[0]
    intercept = regression_model.intercept_

    # Calculate residuals
    residuals = []
    for pred, true in zip(predictions, y_test):
        residuals.append(abs(pred - true))
    
    # Calculate total aggregated residual
    total_residual = sum(residuals)

    mlflow.log_metric("R2 Score", r2)
    mlflow.log_metric("Mean Absolute Error", mae)
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("Gradient", gradient)
    mlflow.log_metric("Intercept", intercept)
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("Total Aggregated Residual", total_residual)

    return gradient, intercept, r2, mape

