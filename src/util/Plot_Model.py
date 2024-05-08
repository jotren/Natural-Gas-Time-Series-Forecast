import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import mlflow
import os

def plot_charts(predictions, y_test, days, gradient, intercept, r2, run_id=None):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Time Series", "Scatter Plot"))

    # Plot time series
    fig.add_trace(go.Scatter(x=days, y=y_test, mode='lines', name='Actual'), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=predictions, mode='lines', name='Predictions'), row=1, col=1)

    # Plot scatter plot
    fig.add_trace(go.Scatter(x=predictions, y=y_test, mode='markers', name='Correlation', marker=dict(color='RoyalBlue')), row=1, col=2)

    # Add a line of best fit to scatter plot
    x_range = np.linspace(0, max(predictions), num=100)
    y_range = [gradient * x + intercept for x in x_range]
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Line of Best Fit', line=dict(color='red')), row=1, col=2)

    # Add equation of the line to scatter plot
    equation_text = f'y = {round(float(gradient), 2)}x + {round(float(intercept), 2)}'
    fig.add_annotation(text=equation_text, x=max(x_range) * 0.4, y=max(y_range), showarrow=False, font=dict(color='red'), row=1, col=2)

    # Add R2 value to scatter plot
    r2_text = f'R^2 = {round(r2, 2)}'
    fig.add_annotation(text=r2_text, x=max(x_range) * 0.4, y=max(y_range) - (max(y_range) - min(y_range)) * 0.1, showarrow=False, font=dict(color='blue'), row=1, col=2)

    fig.update_xaxes(range=[0, max(predictions) * 1.1], row=1, col=2)
    fig.update_yaxes(range=[min(0, min(y_test)) * 0.9, max(y_test) * 1.1], row=1, col=2)
    
    # Update layout
    fig.update_layout(title='Time Series and Scatter Plot of Actual vs Predictions')

    # Save the figure to HTML
    file_path = "plots"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = os.path.join(file_path, "time_series_plot.html")
    fig.write_html(file_name)

    # Check if a run_id is specified, and use the active run context
    if run_id:
        # Ensure we are using the active run context to log the artifact
        mlflow.log_artifact(file_name, artifact_path="plots")
    else:
        # Start a new run only if there's no active run
        with mlflow.start_run():
            mlflow.log_artifact(file_name, artifact_path="plots")

    # Display the figure
    fig.show()
