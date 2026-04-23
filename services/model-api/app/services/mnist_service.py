import time
from datetime import datetime, timezone

import torch
from app.models.loader import load_model
from prometheus_client import Counter, Histogram # prometheus 로 모니터링


MODEL_NAME = "mnist"
MODEL_VERSION = "v1"

prediction_counter = Counter(
    "mnist_predictions_total",
    "예측 횟수",
    ["digit"],
)
confidence_histogram = Histogram(
    "mnist_confidence_score",
    "예측 신뢰도 분포",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def predict(data: dict) -> dict:
    pixels = data.get("pixels")
    if pixels is None or len(pixels) != 784:
        raise ValueError(f"pixels 필드에 784개의 값이 필요합니다. 길이 오류: {len(pixels)}")

    model = load_model(MODEL_NAME, MODEL_VERSION)

    start = time.time()
    with torch.no_grad():
        x = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)
        model_result = model(x)
        probabilities = torch.exp(model_result).squeeze(0)
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item())
    time_ms = round((time.time() - start) * 1000, 3)

    prediction_counter.labels(digit=str(predicted_class)).inc()
    confidence_histogram.observe(confidence)

    return {
        "result": {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
        },
        "metadata": {
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "inference_time_ms": time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
