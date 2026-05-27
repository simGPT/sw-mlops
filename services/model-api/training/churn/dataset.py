import os
import pandas as pd
from sklearn.model_selection import train_test_split

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
TARGET = 'churned'

# 데이터 불러오는 함수
def load_data(data_dir):
    features = pd.read_csv(os.path.join(data_dir, 'ecommerce_customer_features.csv'))
    targets = pd.read_csv(os.path.join(data_dir, 'ecommerce_customer_targets.csv'))
    df = features.merge(targets, on='Customer_ID')
    df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0})
    return df

# 데이터 전처리 함수
def preprocess(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    x = df[FEATURES]
    y = df[TARGET]

    x_train, x_remaining, y_train, y_remaining = train_test_split(
        x, y, test_size=1 - train_ratio, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_remaining, y_remaining, test_size=test_ratio / (val_ratio + test_ratio), random_state=42, stratify=y_remaining
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
