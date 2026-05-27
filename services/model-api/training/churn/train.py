"""
고객 이탈 예측 모델 학습

# 학습 실행
python3 training/churn/train.py --version v1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from training.churn.dataset import load_data, preprocess, FEATURES
from training.churn.evaluate import evaluate


def main(args):
    df = load_data(args.data_dir)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test), _ = preprocess(df)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("churn")

    with mlflow.start_run(run_name=args.version):
        model = LogisticRegression(C=args.C, random_state=42, max_iter=1000)
        model.fit(x_train, y_train)

        val_metrics = evaluate(model, x_valid, y_valid)
        test_metrics = evaluate(model, x_test, y_test)

        mlflow.log_param("version", args.version)
        mlflow.log_param("C", args.C)
        mlflow.log_param("features", FEATURES)
        mlflow.log_metric("valid_f1", val_metrics['f1'])
        mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
        mlflow.log_metric("test_f1", test_metrics['f1'])
        mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"churn-{args.version}",
        )

        print(f"valid_f1    : {val_metrics['f1']:.4f}")
        print(f"test_accuracy: {test_metrics['accuracy']:.4f}")
        print(f"test_f1     : {test_metrics['f1']:.4f}")
        print(f"test_roc_auc: {test_metrics['roc_auc']:.4f}")
        print(f'MLflow에 모델 등록 완료: churn-{args.version}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--data_dir', type=str, default='../data/churn')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5000')
    args = parser.parse_args()

    main(args)
