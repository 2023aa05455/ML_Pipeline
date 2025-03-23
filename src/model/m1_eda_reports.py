import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import os
from ydata_profiling import ProfileReport  # Use ydata-profiling for EDA


def load_fashion_mnist():
    """This function is to Load Fashion MNIST dataset as DataFrame."""
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train_normalized = X_train.astype('float32') / 255.0
    X_test_normalized = X_test.astype('float32') / 255.0

    # Reduce the number of features by taking a smaller subset of pixels (e.g., only a few rows of pixels)
    # We'll select the top-left 14x14 pixels instead of using all 28x28 pixels
    X_train_reduced = X_train_normalized[:, :14, :14]
    X_test_reduced = X_test_normalized[:, :14, :14]

    # Flatten the reduced images
    X_train_flat = X_train_reduced.reshape(X_train_reduced.shape[0], -1)
    X_test_flat = X_test_reduced.reshape(X_test_reduced.shape[0], -1)

    # Convert the data into the DataFrame
    columns = [f'pixel_{i}' for i in range(X_train_flat.shape[1])]
    df_train = pd.DataFrame(X_train_flat, columns=columns)
    df_train['label'] = y_train.astype(int)

    df_test = pd.DataFrame(X_test_flat, columns=columns)
    df_test['label'] = y_test.astype(int)

    return df_train, df_test

def get_key_insights(df):
    """Extract key insights from the dataset."""
    data_insights = {}

    # Verify Class Distribution
    class_counts = df['label'].value_counts(normalize=True) * 100
    imbalance = class_counts.max() - class_counts.min()
    data_insights['class_distribution'] = f"Class imbalance: {imbalance:.2f}%"

    # Find if any Missing Values are there
    missing_values = df.isnull().sum().sum()
    data_insights['missing_values'] = f"Total missing values found are: {missing_values}"

    # Identify Low-Variance Features
    feature_variances = df.drop(columns=['label']).var()
    low_variance_features = feature_variances[feature_variances < 1.0].index.tolist()
    data_insights['low_variance'] = f"Low variance features are : {len(low_variance_features)}"

    # Identify Highly Correlated Features
    corr_matrix = df.drop(columns=['label']).corr().abs()
    high_corr_features = [(i, j, corr_matrix.loc[i, j]) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.9]
    data_insights['high_correlation'] = f"Highly correlated features are: {len(high_corr_features)}"

    return data_insights

def generate_eda_report(df, output_path, sample_size=100):  # Reduced sample size to 100
    """This function is used to Generate an EDA report."""

    df_sample = df.sample(n=min(len(df), sample_size), random_state=42)

    # Ensure 'label' is numeric
    df_sample['label'] = df_sample['label'].astype(int)

    # Generate EDA Report using ydata-profiling with minimal output and more optimizations
    profile = ProfileReport(
        df_sample,
        title="******EDA Report for Fashion MNIST Dataset ******",
        explorative=True,
        minimal=True,
        correlations=None,  # Disable correlation calculations
        missing_diagrams=None,  # Disable missing value diagrams
        interactions=None  # Disable interaction plots
    )
    profile.to_file(output_path)
    print(f"EDA reports are saved at: {output_path}")

def main():
    """Main function to generate EDA reports and extract important insights for the dataset"""
    df_train, df_test = load_fashion_mnist()

    output_dir = "eda_reports"
    os.makedirs(output_dir, exist_ok=True)

    train_report_path = os.path.join(output_dir, "train_fashion_mnist_eda_report.html")
    test_report_path = os.path.join(output_dir, "test_fashion_mnist_eda_report.html")

    print("********* Generating EDA reports for Fashion MNIST dataset *******")
    generate_eda_report(df_train, train_report_path)
    generate_eda_report(df_test, test_report_path)

    # Generate key insights
    data_insights = get_key_insights(df_train)
    # print("\n🔎 Identified the Following Key Insights:")
    for key, value in data_insights.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
