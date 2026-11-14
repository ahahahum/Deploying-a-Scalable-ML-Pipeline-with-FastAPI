import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """The full census data, loaded once for all tests."""
    return pd.read_csv("data/census.csv")


@pytest.fixture(scope="module")
def processed(data):
    """Train/test split plus processed arrays, matching train_model.py."""
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, train, test


def test_train_model_uses_expected_algorithm():
    """train_model should return a random forest classifier."""
    rng = np.random.RandomState(0)
    X = rng.rand(50, 4)
    y = (X[:, 0] > 0.5).astype(int)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_expected_type_and_size():
    """inference should return one 0/1 prediction per input row."""
    rng = np.random.RandomState(0)
    X = rng.rand(50, 4)
    y = (X[:, 0] > 0.5).astype(int)
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_expected_values():
    """Metrics on a small known example should match hand-computed values."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_split_sizes_and_columns(data, processed):
    """The split should be 80/20 and keep the label and feature columns."""
    *_, train, test = processed
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)
    assert "salary" in train.columns and "salary" in test.columns
    assert set(CAT_FEATURES).issubset(train.columns)


def test_processed_data_shapes_and_labels(processed):
    """Train and test should have the same width and binary labels."""
    X_train, y_train, X_test, y_test, train, test = processed
    assert X_train.shape[0] == len(y_train) == len(train)
    assert X_test.shape[0] == len(y_test) == len(test)
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}
