import os
import pandas as pd
from tpot import TPOTClassifier
import xgboost as xgb
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging
import joblib

# Configure logging
logging.basicConfig(filename="automl_results.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Create reports and model save directories
model_hyper_param_reports_dir = "../../model_selection_hyperparameter_reports"
best_model_path_dir = "../../app"


def log_results(message):
    """Log messages to both console and file."""
    print(message)
    logging.info(message)


def save_html_report(df, filename):
    """Save pandas DataFrame as an HTML report."""
    filepath = os.path.join(model_hyper_param_reports_dir, filename)
    df.to_html(filepath, escape=False)  # Allow HTML formatting
    log_results(f"Report saved at: {filepath}")


def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test, y_train, y_test = X_train[:1000], X_test[:300], y_train[:1000], y_test[:300]

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def model_selection_tpot(X_train, y_train, X_val, y_val):
    tpot = TPOTClassifier(generations=3, population_size=10, random_state=42, max_time_mins=30)
    x_train_small, y_train_small = X_train[:1000], y_train[:1000]
    tpot.fit(x_train_small, y_train_small)
    accuracy = accuracy_score(y_val, tpot.predict(X_val))
    log_results(f"TPOT Model Accuracy: {accuracy:.4f}")

    return "TPOT", accuracy, tpot


def model_selection_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_val, model.predict(X_val))
    log_results(f"XGBoost Model Accuracy: {accuracy:.4f}")

    return "XGBoost", accuracy, model


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    log_results("Starting Model Selection...")
    models = [model_selection_tpot, model_selection_xgboost]
    results_data = []
    best_model_name, best_accuracy, best_model = None, 0, None

    for model_func in models:
        model_name, accuracy, model = model_func(X_train, y_train, X_val, y_val)
        results_data.append([model_name, accuracy])
        if accuracy > best_accuracy:
            best_model_name, best_accuracy, best_model = model_name, accuracy, model

    results_df = pd.DataFrame(results_data, columns=["Model", "Accuracy"])

    # Highlight the best model in bold
    results_df["Model"] = results_df.apply(
        lambda row: f"<b>{row['Model']}</b>" if row["Accuracy"] == best_accuracy else row["Model"], axis=1)

    save_html_report(results_df, "model_selection_automl_results.html")

    # Save the best model
    best_model_path = os.path.join(best_model_path_dir, f"best_model_{best_model_name}.pkl")
    joblib.dump(best_model, best_model_path)
    log_results(f"Best model saved at: {best_model_path}")

    log_results("All experiments completed successfully.")


if __name__ == "__main__":
    main()
