import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from src.data import load_data


EXPERIMENT_NAME = "Wine-Cultivar-Classification"


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
        "macro_f1": f1_score(
            y,
            predictions,
            average="macro"
        ),
        "accuracy": accuracy_score(
            y,
            predictions
        ),
        "log_loss": log_loss(
            y,
            probabilities
        )
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

        model.fit(
            X_fold_train,
            y_fold_train
        )

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
        "train_macro_f1": float(
            np.mean(
                [
                    x["macro_f1"]
                    for x in train_scores
                ]
            )
        ),
        "validation_macro_f1": float(
            np.mean(
                [
                    x["macro_f1"]
                    for x in validation_scores
                ]
            )
        ),
        "train_accuracy": float(
            np.mean(
                [
                    x["accuracy"]
                    for x in train_scores
                ]
            )
        ),
        "validation_accuracy": float(
            np.mean(
                [
                    x["accuracy"]
                    for x in validation_scores
                ]
            )
        ),
        "train_log_loss": float(
            np.mean(
                [
                    x["log_loss"]
                    for x in train_scores
                ]
            )
        ),
        "validation_log_loss": float(
            np.mean(
                [
                    x["log_loss"]
                    for x in validation_scores
                ]
            )
        )
    }

    return results


def run_experiment(
    model_family,
    model_class,
    configuration,
    params,
    X_train,
    y_train
):

    with mlflow.start_run() as run:

        mlflow.set_tag(
            "model_family",
            model_family
        )

        mlflow.set_tag(
            "configuration",
            configuration
        )

        mlflow.log_params(params)

        metrics = cross_validate_model(
            model_class,
            params,
            X_train,
            y_train
        )

        mlflow.log_metrics(metrics)

        final_model = model_class(
            random_state=42,
            **params
        )

        final_model.fit(
            X_train,
            y_train
        )

        input_example = X_train.head(5)

        signature = mlflow.models.infer_signature(
            X_train,
            final_model.predict(X_train)
        )

        logged_model = mlflow.sklearn.log_model(
            final_model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print()
        print("Model:", model_family)
        print("Configuration:", configuration)
        print("Parameters:", params)
        print("Train Macro F1:", metrics["train_macro_f1"])
        print(
            "Validation Macro F1:",
            metrics["validation_macro_f1"]
        )
        print(
            "Train Accuracy:",
            metrics["train_accuracy"]
        )
        print(
            "Validation Accuracy:",
            metrics["validation_accuracy"]
        )
        print(
            "Train Log Loss:",
            metrics["train_log_loss"]
        )
        print(
            "Validation Log Loss:",
            metrics["validation_log_loss"]
        )
        print("Run ID:", run.info.run_id)
        print(
            "Logged Model ID:",
            logged_model.model_id
        )

        return {
            "run_id": run.info.run_id,
            "logged_model_id": logged_model.model_id,
            "model_family": model_family,
            "configuration": configuration,
            "parameters": params,
            **metrics
        }


def run_experiments():

    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

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

    for i, params in enumerate(
        rf_configs,
        1
    ):

        result = run_experiment(
            "RandomForestClassifier",
            RandomForestClassifier,
            i,
            params,
            X_train,
            y_train
        )

        results.append(result)

    for i, params in enumerate(
        gb_configs,
        1
    ):

        result = run_experiment(
            "GradientBoostingClassifier",
            GradientBoostingClassifier,
            i,
            params,
            X_train,
            y_train
        )

        results.append(result)

    return results


if __name__ == "__main__":
    run_experiments()
