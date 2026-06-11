import numpy as np

# 모델 평가 함수
def evaluate(model, X, y, treatment):
    treatment = np.array(treatment)
    uplift_scores = model.predict(X) # uplift 점수

    treatment_outcome_rate = y[treatment == 1].mean() # treatment 그룹의 재방문율
    control_outcome_rate = y[treatment == 0].mean() # control 그룹의 재방문율
    avg_uplift = uplift_scores.mean() # 전체 샘플의 평균 uplift 점수

    high_uplift_mask = uplift_scores > np.percentile(uplift_scores, 75) # uplift 점수가 상위 25%인 샘플
    high_uplift_treatment_rate = y[(high_uplift_mask) & (treatment == 1)].mean() if (high_uplift_mask & (treatment == 1)).sum() > 0 else 0 # 실제 uplift 점수가 높은 샘플 중 treatment 그룹의 재방문율(uplift 점수 vs outcome)
    high_uplift_control_rate = y[(high_uplift_mask) & (treatment == 0)].mean() if (high_uplift_mask & (treatment == 0)).sum() > 0 else 0 # 실제 uplift 점수가 높은 샘플 중 control 그룹의 재방문율(uplift 점수 vs outcome)

    return {
        "treatment_outcome_rate": float(treatment_outcome_rate),
        "control_outcome_rate": float(control_outcome_rate),
        "avg_uplift_score": float(avg_uplift),
        "high_uplift_treatment_rate": float(high_uplift_treatment_rate),
        "high_uplift_control_rate": float(high_uplift_control_rate),
    }
