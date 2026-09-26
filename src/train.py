import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from src.data import load_data


def train_random_forest(params, X_train, y_train):
    model = RandomForestClassifier(
        random_state=42,
        **params
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(params, X_train, y_train):
    model = GradientBoostingClassifier(
        random_state=42,
        **params
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return {
        "macro_f1": f1_score(y, predictions, average="macro"),
        "accuracy": accuracy_score(y, predictions),
        "log_loss": log_loss(y, probabilities)
    }


def cross_validate_model(model_class, params, X_train, y_train):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_scores = []
    validation_scores = []

    for train_index, validation_index in skf.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_index]
        X_fold_validation = X_train.iloc[validation_index]
        y_fold_train = y_train.iloc[train_index]
        y_fold_validation = y_train.iloc[validation_index]

        model = model_class(
            random_state=42,
            **params
        )

        model.fit(X_fold_train, y_fold_train)

        train_metrics = evaluate_model(
            model,
            X_fold_train,
            y_fold_train
        )

        validation_metrics = evaluate_model(
            model,
            X_fold_validation,
            y_fold_validation
        )

        train_scores.append(train_metrics)
        validation_scores.append(validation_metrics)

    results = {
        "train_macro_f1": np.mean([x["macro_f1"] for x in train_scores]),
        "validation_macro_f1": np.mean(
            [x["macro_f1"] for x in validation_scores]
        ),
        "train_accuracy": np.mean([x["accuracy"] for x in train_scores]),
        "validation_accuracy": np.mean(
            [x["accuracy"] for x in validation_scores]
        ),
        "train_log_loss": np.mean([x["log_loss"] for x in train_scores]),
        "validation_log_loss": np.mean(
            [x["log_loss"] for x in validation_scores]
        )
    }

    return results


def run_experiments():
    X_train, X_test, y_train, y_test = load_data()

    rf_configs = [
        {
            "n_estimators": 100,
            "max_depth": 5
        },
        {
            "n_estimators": 200,
            "max_depth": 10
        },
        {
            "n_estimators": 300,
            "max_depth": None
        }
    ]

    gb_configs = [
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 2
        },
        {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 3
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 4
        }
    ]

    results = []

    for i, params in enumerate(rf_configs, 1):
        metrics = cross_validate_model(
            RandomForestClassifier,
            params,
            X_train,
            y_train
        )

        results.append({
            "model": "RandomForestClassifier",
            "configuration": i,
            "parameters": params,
            **metrics
        })

    for i, params in enumerate(gb_configs, 1):
        metrics = cross_validate_model(
            GradientBoostingClassifier,
            params,
            X_train,
            y_train
        )

        results.append({
            "model": "GradientBoostingClassifier",
            "configuration": i,
            "parameters": params,
            **metrics
        })

    for result in results:
        print(result)

    return results


if __name__ == "__main__":
    run_experiments()
