import time
from datetime import datetime, timezone

import torch
from app.models.loader import load_model


MODEL_NAME = "mnist"
MODEL_VERSION = "v1"


def predict(data: dict) -> dict:
    pixels = data.get("pixels") # pixels 키에서 꺼낸 값이 784아니면 이상
    if pixels is None or len(pixels) != 784:
        raise ValueError(f"pixels 필드에 784개의 값이 필요합니다. 길이 오류: {len(pixels)}")

    model = load_model(MODEL_NAME, MODEL_VERSION)

    start = time.time()
    with torch.no_grad(): # 학습이 아님 -> 가중치 계산 뺌
        x = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)  # 입력값 텐서플로우로 변경
        model_result = model(x) # 모델에 입력 후 결과
        probabilities = torch.exp(model_result).squeeze(0) # 결과 정리
        predicted_class = int(torch.argmax(probabilities).item()) # 가장 확률이 높은 인덱스
        confidence = float(probabilities[predicted_class].item()) # 해당 인덱스의 값
    time_ms = round((time.time() - start) * 1000, 3) # 예측 시간 계산

    return {
        "result": {
            "predicted_class": predicted_class, # 예측 숫자 
            "confidence": confidence,   # 확률
            "probabilities": probabilities.tolist(),
        },
        "metadata": {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "inference_time_ms": time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
