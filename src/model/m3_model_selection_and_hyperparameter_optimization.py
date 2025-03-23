import os
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from tpot import TPOTClassifier
import logging

# Configure logging
logging.basicConfig(filename="automl_results.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Create reports directory
# os.makedirs("model_hyperparameter_reports", exist_ok=True)
model_hyper_param_reports_dir = "../../model_selection_hyperparameter_reports"

def log_results(message):
    """Log messages to both console and file."""
    print(message)
    logging.info(message)


def save_html_report(df, filename):
    """Save pandas DataFrame as an HTML report."""
    filepath = os.path.join(model_hyper_param_reports_dir, filename)
    df.to_html(filepath)
    log_results(f"Report are saved at: {filepath}")


def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_test, y_train, y_test = X_train[:1000], X_test[:300], y_train[:1000], y_test[:300]

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def model_selection_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_val, model.predict(X_val))
    log_results(f"XGBoost Baseline Model Accuracy: {accuracy:.4f}")

    results_df = pd.DataFrame({"Model": ["XGBoost"], "Accuracy": [accuracy]})
    save_html_report(results_df, "model_selection_automl_results.html")

    return model


def model_trials(trial, X_train, y_train, X_val, y_val):
    n_estimators = trial.suggest_int("n_estimators", 50, 150, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 12, step=3)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, step=0.05)

    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                              random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)

    log_results(
        f"Trial {trial.number}: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, accuracy={accuracy:.4f}")
    return accuracy


def hyperparameter_tuning_optuna(X_train, y_train, X_val, y_val):
    tuning_results = optuna.create_study(direction='maximize')
    tuning_results.optimize(lambda trial: model_trials(trial, X_train, y_train, X_val, y_val), n_trials=5)

    best_params = tuning_results.best_params
    log_results(f"Best Hyperparameters are : {best_params}")

    trials_df = tuning_results.trials_dataframe()
    save_html_report(trials_df, "hyperparameter_tuning_results.html")

    return best_params


def model_selection_tpot(X_train, y_train, X_val, y_val):
    tpot = TPOTClassifier(generations=3, population_size=10, random_state=42, max_time_mins=30, n_jobs=1)
    x_train_small, y_train_small = X_train[:1000], y_train[:1000]
    tpot.fit(x_train_small, y_train_small)
    validation_accuracy = tpot.fitted_pipeline_.score(X_val, y_val)
    log_results(f"TPOT Best Model Accuracy: {validation_accuracy:.4f}")
    # tpot.export('best_model.py')

    results_df = pd.DataFrame({"Model": ["TPOT"], "Accuracy": [validation_accuracy]})
    save_html_report(results_df, "topt_results.html")


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    log_results("Starting XGBoost Model Selection...")
    model_selection_xgboost(X_train, y_train, X_val, y_val)

    log_results("Starting Optuna Hyperparameter Tuning...")
    best_params_xgb = hyperparameter_tuning_optuna(X_train, y_train, X_val, y_val)

    log_results("Starting TPOT AutoML...")
    model_selection_tpot(X_train, y_train, X_val, y_val)

    log_results("All experiments completed successfully.")


if __name__ == "__main__":
    main()
