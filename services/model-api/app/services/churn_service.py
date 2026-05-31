import os
import time
from datetime import datetime, timezone

from app.models.loader import load_churn_model
from prometheus_client import Counter, Histogram

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

prediction_counter = Counter(
    "churn_predictions_total",
    "이탈 예측 횟수",
    ["predicted"],
)
confidence_histogram = Histogram(
    "churn_confidence_score",
    "이탈 예측 신뢰도 분포",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def predict(data: dict) -> dict:
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"누락된 피처: {missing}")

    model = load_churn_model(MODEL_NAME)

    x = [[data[f] for f in FEATURES]]

    start = time.time()
    predicted = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    confidence = float(proba[predicted])
    time_ms = round((time.time() - start) * 1000, 3)

    prediction_counter.labels(predicted=str(predicted)).inc()
    confidence_histogram.observe(confidence)

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
