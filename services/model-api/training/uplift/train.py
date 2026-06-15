"""
Uplift 모델 학습 (T-Learner)

# 학습 실행
python3 training/uplift/train.py --version v1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mlflow
import mlflow.sklearn
import numpy as np
import requests
from sklearn.model_selection import train_test_split

from app.models.uplift_model import TLearner
from training.uplift.dataset import generate_dummy_data, load_real_data, FEATURES
from training.uplift.evaluate import evaluate


def main(args):
    if args.data_path:
        df = load_real_data(args.data_path)
    else:
        df = generate_dummy_data(n_samples=5000)

    X = df[FEATURES].values
    y = df['outcome'].values
    treatment = df['treatment'].values

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, treatment, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("uplift")

    with mlflow.start_run(run_name=args.version):
        model = TLearner(C=args.C)
        model.fit(X_train, y_train, t_train) # 모델 학습

        metrics = evaluate(model, X_test, y_test, t_test) # 모델 평가

        # mlflow 파라미터 저장
        mlflow.log_param("version", args.version) # 모델 버전
        mlflow.log_param("C", args.C) # 모델 하이퍼파라미터
        mlflow.log_param("n_samples", len(df)) # 학습에 사용된 데이터 샘플 수
        mlflow.log_param("features", FEATURES) # 모델에 사용된 피처 목록

        # mlflow 메트릭 저장
        mlflow.log_metric("treatment_outcome_rate", metrics["treatment_outcome_rate"]) # treatment 그룹의 재방문율
        mlflow.log_metric("control_outcome_rate", metrics["control_outcome_rate"]) # control 그룹의 재방문율
        mlflow.log_metric("avg_uplift_score", metrics["avg_uplift_score"]) # 전체 샘플의 평균 uplift 점수
        mlflow.log_metric("high_uplift_treatment_rate", metrics["high_uplift_treatment_rate"]) # 실제 uplift 점수가 높은 샘플 중 treatment 그룹의 재방문율(uplift 점수 vs outcome)
        mlflow.log_metric("high_uplift_control_rate", metrics["high_uplift_control_rate"]) # 실제 uplift 점수가 높은 샘플 중 control 그룹의 재방문율

        # mlflow 모델 저장 및 등록
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="uplift",
        )

        print(f"treatment_outcome_rate : {metrics['treatment_outcome_rate']:.4f}")
        print(f"control_outcome_rate   : {metrics['control_outcome_rate']:.4f}")
        print(f"avg_uplift_score       : {metrics['avg_uplift_score']:.4f}")
        print(f"MLflow에 모델 등록 완료: uplift-{args.version}")

    # 배포 트리거
    github_token = os.getenv("GITHUB_TOKEN")
    github_owner = os.getenv("GITHUB_REPO_OWNER")
    github_repo = os.getenv("GITHUB_REPO_NAME")

    if github_token and github_owner and github_repo:
        response = requests.post(
            f"https://api.github.com/repos/{github_owner}/{github_repo}/dispatches",
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github+json",
            },
            json={"event_type": "deploy-uplift"},
        )
        if response.status_code == 204:
            print("배포 트리거 완료")
        else:
            print(f"배포 트리거 실패: {response.status_code}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--mlflow_uri', type=str, default='https://mlflow.swmlops.site')
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()

    main(args)
