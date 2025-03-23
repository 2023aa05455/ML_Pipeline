#-------------- Fetch MLFLOW experiments---------------
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import pandas as pd
import os
mlruns_dir = "mlruns"
os.environ['MLFLOW_TRACKING_URI'] = mlruns_dir
mlflow.set_tracking_uri('../../mlruns')
# Initialize MLflow client
client = MlflowClient()
# current_dir = os.getcwd()
# print(current_dir)
# Get Experiment ID (replace with your actual experiment name)
experiment_name = "Fashion_MNIST_Model"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Fetch all runs from the experiment
runs = client.search_runs(experiment_id, order_by=["metrics.test_accuracy DESC"])

# Convert to Pandas DataFrame for analysis
df = pd.DataFrame([{
    "run_id": run.info.run_id,
    "test_accuracy": run.data.metrics["accuracy"],
    "start_time": run.info.start_time,
    "duration": run.info.end_time - run.info.start_time

} for run in runs])

# Convert start_time to datetime format for better visualization
df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
pd.set_option('display.width', None)  # This removes the width limit for display

# Apply styling to center both headers and data
df_styled = df.style.set_table_styles([
    {'selector': 'th', 'props': [('text-align', 'center')]},  # Center headers
    {'selector': 'td', 'props': [('text-align', 'center')]},  # Center data cells
])

html = df_styled.to_html()
print(html)

# Save the HTML to a file to view in the browser
with open('../../model_prerformance_results/Model_Performances_Results.html', 'w') as file:
    file.write(html)

# It will open the HTML file in a browser
import webbrowser
webbrowser.open('Model Performances Results.html')
