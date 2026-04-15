import os
import torch
from app.models.mnist_model import get_model


_model_cache: dict = {}


def load_model(model_name: str, version: str, artifact_dir: str = "artifacts"):
    cache_key = f"{model_name}_{version}" # 캐시 있으면 그거 먼저 확인
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    path = os.path.join(artifact_dir, f"model_{version}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일 없음: {path}")

    if model_name == "mnist":
        model = get_model()
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    _model_cache[cache_key] = model
    return model
