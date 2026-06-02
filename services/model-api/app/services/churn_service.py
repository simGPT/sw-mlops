import os
import time
from datetime import datetime, timezone

from app.models.loader import load_churn_model
from prometheus_client import Counter, Gauge, Histogram

MODEL_NAME = "churn"

FEATURES = [
    'account_age_months',
    'avg_order_value',
    'total_orders',
    'days_since_last_purchase',
    'discount_usage_rate',
    'return_rate',
    'browsing_frequency_per_week',
    'cart_abandonment_rate',
]

# Prometheus 메트릭 정의
# 총 예측 횟수, 예측된 클래스별로 레이블(predicted) 추가
prediction_counter = Counter( # Counter는 총 횟수를 세는 메트릭
    "churn_predictions_total",
    "이탈 예측 횟수",
    ["predicted"],
)
# 예측 신뢰도 분포
confidence_histogram = Histogram( # Histogram은 값의 분포를 나타내는 메트릭
    "churn_confidence_score",
    "이탈 예측 신뢰도 분포",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
# 입력 피처의 평균값
feature_mean_gauge = Gauge( # Gauge는 현재 값을 나타내는 메트릭
    "churn_feature_mean",
    "입력 피처 평균값",
    ["feature"],
)
# 추론 지연 시간
inference_histogram = Histogram(
    "churn_inference_duration_seconds",
    "추론 지연 시간",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)


def predict(data: dict) -> dict:
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"누락된 피처: {missing}")

    model = load_churn_model(MODEL_NAME)

    x = [[data[f] for f in FEATURES]]

    for feature in FEATURES:
        feature_mean_gauge.labels(feature=feature).set(data[feature])

    start = time.time()
    predicted = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    confidence = float(proba[predicted])
    duration = time.time() - start
    time_ms = round(duration * 1000, 3)

    prediction_counter.labels(predicted=str(predicted)).inc()
    confidence_histogram.observe(confidence)
    inference_histogram.observe(duration)

    return {
        "result": {
            "predicted_class": predicted,
            "churned": bool(predicted),
            "confidence": confidence,
            "probabilities": {"retained": float(proba[0]), "churned": float(proba[1])},
        },
        "metadata": {
            "model": MODEL_NAME,
            "inference_time_ms": time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
