import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"


def register_champion():
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError("Experiment not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.validation_macro_f1 DESC"]
    )

    if not runs:
        raise ValueError("No MLflow runs found.")

    best_run = runs[0]

    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics["validation_macro_f1"]

    print(f"Best run: {best_run_id}")
    print(f"Best validation Macro F1: {best_f1}")

    logged_models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        output_format="list"
    )

    best_model = None

    for model in logged_models:
        if model.source_run_id == best_run_id:
            best_model = model
            break

    if best_model is None:
        raise ValueError(
            f"No logged model found for best run {best_run_id}."
        )

    print(f"Logged Model ID: {best_model.model_id}")

    model_uri = f"models:/{best_model.model_id}"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Registered model: {registered_model.name}")
    print(f"Model version: {registered_model.version}")

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered_model.version
    )

    print("Alias: champion")


if __name__ == "__main__":
    register_champion()
