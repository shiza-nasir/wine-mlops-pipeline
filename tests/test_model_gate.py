import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import load_data


def test_metric_threshold():
    X_train, _, y_train, _ = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv
    )

    validation_f1 = f1_score(
        y_train,
        predictions,
        average="macro"
    )

    assert validation_f1 >= 0.88


def test_inference_latency():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    start_time = time.perf_counter()

    model.predict(X_test)

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    assert latency_ms <= 30


def test_output_schema():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    valid_classes = {0, 1, 2}

    assert set(np.unique(predictions)).issubset(valid_classes)
