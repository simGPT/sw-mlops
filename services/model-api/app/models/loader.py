import os
import mlflow.pytorch
import mlflow.sklearn


_model_cache: dict = {}

# mlflow에서 mnist 모델 로드하는 함수
def load_model(model_name: str):
    cache_key = model_name
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000") # 환경변수에서 mlflow tracking uri 가져오기, 없으면 기본값으로 http://mlflow:5000 사용
    mlflow.set_tracking_uri(mlflow_uri)

    model_uri = f"models:/{model_name}/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    _model_cache[cache_key] = model
    return model

# mlflow에서 고객 이탈 예측 모델 로드하는 함수
def load_churn_model(model_name: str):
    cache_key = f"{model_name}_sklearn"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000") # 환경변수에서 mlflow tracking uri 가져오기, 없으면 기본값으로 http://mlflow:5000 사용
    mlflow.set_tracking_uri(mlflow_uri)

    model_uri = f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)

    _model_cache[cache_key] = model
    return model
