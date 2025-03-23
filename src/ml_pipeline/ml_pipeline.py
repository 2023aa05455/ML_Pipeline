from perfect import task

@task
def eda_report_generation():
    """Generate EDA reports."""
    import os
    os.system("python m1_eda_reports.py")
    return ["eda_reports/train_fashion_mnist_eda_report.html", "eda_reports/test_fashion_mnist_eda_report.html"]

@task
def feature_engineering_and_explainability(eda_outputs):
    """Perform feature engineering and explainability analysis."""
    import os
    os.system("python m2_feature_engineering_and_explainability.py")
    return ["feature_engg_explainability_reports/shap_summary_plot.png", "feature_engg_explainability_reports/shap_feature_importance.csv"]

@task
def model_selection_and_hyperparameter_tuning(feature_outputs):
    """Select model and optimize hyperparameters."""
    import os
    os.system("python m3_model_selection_and_hyperparameter_optimization.py")
    return ["model_selection_hyperparameter_reports/model_selection_automl_results.html", "model_selection_hyperparameter_reports/hyperparameter_tuning_results.html"]

@task
def model_experimentation_and_tracking(model_selection_outputs):
    """Track model experimentation and performance."""
    import os
    os.system("python m4_model_experimentation_tracking.py")
    return ["model_tracking_reports/model_accuracy_over_time.png", "mlruns"]

def main():
    """Execute the MLOps pipeline sequentially."""
    eda_outputs = eda_report_generation()
    feature_outputs = feature_engineering_and_explainability(eda_outputs)
    model_selection_outputs = model_selection_and_hyperparameter_tuning(feature_outputs)
    model_experimentation_and_tracking(model_selection_outputs)

if __name__ == "__main__":
    main()
