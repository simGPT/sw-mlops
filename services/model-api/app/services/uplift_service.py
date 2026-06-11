import time
from datetime import datetime, timezone

from app.models.loader import load_uplift_model

MODEL_NAME = "uplift"

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


def predict(data: dict) -> dict:
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"누락된 피처: {missing}")

    model = load_uplift_model(MODEL_NAME) # 모델 로드

    x = [[data[f] for f in FEATURES]]

    start = time.time()
    uplift_score = float(model.predict(x)[0])
    duration = time.time() - start

    return {
        "result": {
            "uplift_score": uplift_score,
            "is_persuadable": uplift_score > 0, # 마케팅 액션이 효과가 있는 고객인지 여부
        },
        "metadata": {
            "model": MODEL_NAME,
            "inference_time_ms": round(duration * 1000, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
