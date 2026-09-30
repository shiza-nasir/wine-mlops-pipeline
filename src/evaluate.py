import time

import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from src.data import load_data


MODEL_URI = "models:/WineClassifier@champion"


def evaluate_champion():

    X_train, X_test, y_train, y_test = load_data()

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    start_time = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    end_time = time.perf_counter()

    probabilities = model.predict_proba(
        X_test
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    test_log_loss = log_loss(
        y_test,
        probabilities
    )

    inference_time_ms = (
        end_time - start_time
    ) * 1000

    print(
        "Test Macro F1:",
        macro_f1
    )

    print(
        "Test Accuracy:",
        accuracy
    )

    print(
        "Test Log Loss:",
        test_log_loss
    )

    print(
        "Inference Time (ms):",
        inference_time_ms
    )


if __name__ == "__main__":
    evaluate_champion()
