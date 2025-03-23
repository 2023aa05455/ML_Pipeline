import os
import joblib
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from alibi_detect.cd import KSDrift
from tensorflow.keras.datasets import fashion_mnist

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories
MODEL_RESULTS_DIR = '../../model_prerformance_results'
MODEL_TRACKING_REPORTS_DIR = '../../model_tracking_reports'
MLRUNS_DIR = '../../mlruns'
os.environ['MLFLOW_TRACKING_URI'] = MLRUNS_DIR


def load_and_preprocess_data():
    """Load and preprocess the Fashion MNIST dataset."""
    logging.info("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0

    logging.info("Data preprocessing completed.")
    return x_train, y_train, x_test, y_test


def define_models():
    """Define machine learning models for training."""
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42)
    }


def setup_mlflow(experiment_name="Fashion_MNIST_Model"):
    """Setup MLflow for model tracking."""
    logging.info("Setting up MLflow...")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow experiment '{experiment_name}' is ready.")


def train_and_log_model(model, X_train, y_train, X_test, y_test):
    """Train model and log performance metrics to MLflow."""
    model_name = type(model).__name__
    logging.info(f"Training {model_name}...")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log metrics
        mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'f1_score': f1})
        mlflow.log_param('model_name', model_name)

        # Save model to MLflow
        input_example = X_train[0].reshape(1, -1)
        mlflow.sklearn.log_model(model, 'model', input_example=input_example)

        logging.info(f"Logged {model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        return model, accuracy


def save_model(model, filename='image_classification.joblib'):
    """Save trained model using joblib."""
    joblib.dump(model, f'../../app/{filename}')
    logging.info(f"Model saved as {filename}")


def detect_drift(X_train, X_test):
    """Detect dataset drift using KSDrift from Alibi Detect."""
    drift_detector = KSDrift(X_train, p_val=0.05)
    drift_result = drift_detector.predict(X_test)
    is_drift = drift_result['data'].get('is_drift', False)

    if is_drift:
        logging.warning("Drift detected in test data!")
    else:
        logging.info("No drift detected.")
    return is_drift


def simulate_drift(X_test):
    """Simulate drift by adding noise to test data."""
    noise = np.random.normal(0, 0.1, X_test.shape)
    X_test_drifted = np.clip(X_test + noise, 0, 1)
    logging.info("Drift simulated in test data.")
    return X_test_drifted


def monitor_performance(models, X_train, y_train, X_test, y_test):
    """Monitor model performance over multiple runs and detect drift."""
    accuracies = {model_name: [] for model_name in models}

    for model_name, model in models.items():
        logging.info(f"Training and evaluating {model_name}...")
        model, accuracy = train_and_log_model(model, X_train, y_train, X_test, y_test)
        accuracies[model_name].append(accuracy)

        X_test_drifted = simulate_drift(X_test)
        if detect_drift(X_train, X_test_drifted):
            logging.info(f"Retraining {model_name} due to detected drift...")
            model, accuracy = train_and_log_model(model, X_train, y_train, X_test, y_test)
            save_model(model)
            accuracies[model_name].append(accuracy)

        performance_log = {
            'run_time': datetime.now(),
            'model': model_name,
            'accuracy': accuracy
        }

        pd.DataFrame([performance_log]).to_csv(
            os.path.join(MODEL_RESULTS_DIR, 'performance_log.csv'), mode='a', header=False, index=False)

    # Plot Performance Over Time
    plt.figure(figsize=(10, 6))
    for model_name, model_accuracies in accuracies.items():
        plt.plot(model_accuracies, label=model_name)

    plt.title("Model Accuracy Over Time")
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(MODEL_TRACKING_REPORTS_DIR, 'model_accuracy_over_time.png'))
    plt.close()
    logging.info("Performance monitoring completed.")


def run_ml_pipeline():
    """Execute the complete MLOps pipeline."""
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    models = define_models()
    setup_mlflow()
    monitor_performance(models, X_train, y_train, X_test, y_test)
    logging.info("MLOps pipeline execution completed.")


if __name__ == "__main__":
    run_ml_pipeline()
