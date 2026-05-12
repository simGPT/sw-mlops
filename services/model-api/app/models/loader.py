import os
import mlflow.pytorch


_model_cache: dict = {}


def load_model(model_name: str, version: str):
    cache_key = f"{model_name}_{version}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    model_uri = f"models:/{model_name}-{version}/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    _model_cache[cache_key] = model
    return model
