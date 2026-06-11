import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 각 그룹을 로지스틱 회귀모델로 학습 후 예측값의 차이로 uplift 점수 계산하는 T-learner 모델
class TLearner(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C
        # treatment(사용자에게 프로모션을 제공한 그룹)과 control(프로모션을 제공하지 않은 그룹) 각각에 대해 모델을 학습
        self.treatment_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=C, random_state=42, max_iter=1000)),
        ])
        self.control_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=C, random_state=42, max_iter=1000)),
        ])
    # fit 메서드에서는 treatment 그룹과 control 그룹을 나누어 각각의 모델을 학습
    def fit(self, X, y, treatment):
        treatment = np.array(treatment)
        self.treatment_model.fit(X[treatment == 1], y[treatment == 1])
        self.control_model.fit(X[treatment == 0], y[treatment == 0])
        return self
    # predict 메서드에서는 두 모델의 예측값의 차이로 uplift 점수를 계산하여 반환
    # (마케팅을 하면 구매할 확률) - (마케팅을 안 하면 구매할 확률) = uplift 점수
    def predict(self, X):
        p_treatment = self.treatment_model.predict_proba(X)[:, 1]
        p_control = self.control_model.predict_proba(X)[:, 1]
        return p_treatment - p_control
