import os
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress TensorFlow CPU feature warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations warning

# Suppress specific TensorFlow and Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")


# Load Fashion MNIST
def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Flatten images to 1D vectors
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    return X_train, y_train, X_test, y_test


# Feature Engineering (Normalization and Scaling)
def preprocess_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# Train a simple model for explainability
def train_model(X_train, y_train, epochs=10):
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)
    return model


# Explainability Analysis and Saving SHAP Reports
def explainability_analysis(model, X_train):
    # Ensure X_train is a NumPy array
    X_train_np = np.array(X_train[:1000], dtype=np.float32)

    explainer = shap.GradientExplainer(model, X_train_np)
    shap_values = explainer.shap_values(X_train_np)

    reports_path = "../../feature_engg_explainability_reports"
    os.makedirs(reports_path, exist_ok=True)

    shap_png_path = os.path.join(reports_path, "shap_summary_plot.png")
    shap_features_path = os.path.join(reports_path, "all_shap_values.html")

    # Generate SHAP Summary Plot and Save as PNG
    shap.summary_plot(shap_values, X_train_np, show=False)
    plt.savefig(shap_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved at: {shap_png_path}")

    # Aggregate SHAP values across all classes
    if isinstance(shap_values, list):
        mean_shap_values = np.mean(np.abs(np.array(shap_values)), axis=(0, 1))
    else:
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    mean_shap_values = mean_shap_values.flatten()
    feature_importance = pd.Series(mean_shap_values, index=[f"Feature {i}" for i in range(len(mean_shap_values))])

    # Save all SHAP values as HTML
    feature_importance.to_csv(os.path.join(reports_path, "shap_feature_importance.csv"))
    print(f"All SHAP values saved at: {shap_features_path}")

    # Select a larger number of important features dynamically
    threshold = np.percentile(mean_shap_values, 75)
    top_features_list = [i for i, val in enumerate(mean_shap_values) if val > threshold]

    top_features_list = [idx for idx in top_features_list if idx < X_train.shape[1]]

    return top_features_list


# Main function
def main():
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_fashion_mnist()

    print("Preprocessing data...")
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    print("Training model...")
    model = train_model(X_train_scaled, y_train, epochs=10)

    print("Running explainability analysis...")
    top_feature_indices = explainability_analysis(model, X_train_scaled)

    print(f"\nRefining features: Using top {len(top_feature_indices)} features for model training")
    X_train_refined = X_train_scaled[:, top_feature_indices]
    X_test_refined = X_test_scaled[:, top_feature_indices]

    print("Retraining model with selected features...")
    refined_model = train_model(X_train_refined, y_train, epochs=10)

    test_accuracy = refined_model.evaluate(X_test_refined, y_test, verbose=1)
    print(f"Test Accuracy with refined features: {test_accuracy[1]}")


if __name__ == "__main__":
    main()
