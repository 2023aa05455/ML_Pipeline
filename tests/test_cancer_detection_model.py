import pytest
from mlops.src.model.cancer_detection_model import (
    train_and_evaluate,
    preprocess_data,
    load_data,
    CONFIG,
)


@pytest.fixture
def data():
    """Fixture to load and preprocess data."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    return X_train, X_test, y_train, y_test


def test_train_and_evaluate(data):
    """Test for training and evaluating models."""
    X_train, X_test, y_train, y_test = data

    for model_name, model_config in CONFIG["models"].items():
        model, accuracy, precision, f1 = train_and_evaluate(
            model_name, model_config, X_train, X_test, y_train, y_test
        )
        assert accuracy > 0.9, f"Model accuracy should be >0.9, got {accuracy}"
