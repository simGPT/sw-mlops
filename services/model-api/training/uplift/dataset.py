import numpy as np
import pandas as pd

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

def load_real_data(csv_path: str):
    df = pd.read_csv(csv_path)
    required = FEATURES + ['treatment', 'outcome']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 누락된 컬럼: {missing}")
    return df[required]


# 더미 데이터 생성 함수
def generate_dummy_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    df = pd.DataFrame({
        'account_age_months': np.random.randint(1, 60, n_samples),
        'avg_order_value': np.random.uniform(10000, 200000, n_samples),
        'total_orders': np.random.randint(1, 50, n_samples),
        'days_since_last_purchase': np.random.randint(1, 365, n_samples),
        'discount_usage_rate': np.random.uniform(0, 1, n_samples),
        'return_rate': np.random.uniform(0, 0.5, n_samples),
        'browsing_frequency_per_week': np.random.uniform(0, 20, n_samples),
        'cart_abandonment_rate': np.random.uniform(0, 1, n_samples),
    })

    df['treatment'] = np.random.randint(0, 2, n_samples)

    # 이탈 위험이 높을수록 재방문 확률 낮음 + treatment 그룹에 uplift 효과 추가
    base_prob = (
        0.3
        + 0.2 * (df['total_orders'] / 50)
        - 0.2 * (df['days_since_last_purchase'] / 365)
        - 0.1 * df['cart_abandonment_rate']
    )
    treatment_effect = 0.15 * df['treatment']
    prob = np.clip(base_prob + treatment_effect, 0.05, 0.95)
    df['outcome'] = np.random.binomial(1, prob)

    return df
