"""
로컬 학습용 (Docker 이미지에 포함되지 않음 -> 빌드 시간이 너무 오래걸려서 학습용은 따로 뺌)

의존성 설치:
pip install -r requirements-train.txt

conda 환경 활성화:
conda activate mlops

# v1 초기 학습
python3 training/mnist/train.py --version v1

# v2 재학습 (train + retrain 데이터 합쳐서 학습)
python3 training/mnist/train.py --version v2
"""
import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import mlflow
import mlflow.pytorch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from app.models.mnist_model import get_model
from training.mnist.dataset import load_data, split_data
from training.mnist.trainer import train
from training.mnist.evaluate import evaluate


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 데이터 준비
    x, y, test_raw = load_data(args.data_dir)
    x, y = split_data(x, y, test_raw)
    x = [x_i.to(device) for x_i in x]
    y = [y_i.to(device) for y_i in y]

    # 학습 데이터 결정 -> v1: 첫학습, v2: 재학습
    if args.version == 'v1':
        x_train, y_train = x[0], y[0]
    elif args.version == 'v2':
        x_train = torch.cat([x[0], x[1]], dim=0)
        y_train = torch.cat([y[0], y[1]], dim=0)
    else:
        raise ValueError(f'잘못된 버전입니다: {args.version}')

    # 모델 초기화
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("mnist")

    with mlflow.start_run(run_name=args.version):
        # 모델 학습
        model, _, _ = train(
            x_train, y_train,
            x[2], y[2],  # valid
            model, optimizer,
        )

        # 테스트
        _, test_acc = evaluate(model, x[3], y[3])

        # MLflow에 메트릭 기록
        mlflow.log_param("version", args.version)
        mlflow.log_metric("test_accuracy", test_acc)

        # MLflow에 모델 등록 (S3에 자동 저장)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"mnist-{args.version}",
        )

    print(f'MLflow에 모델 등록 완료: mnist-{args.version}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--mlflow_uri', type=str, default='http://localhost:5000')
    args = parser.parse_args()

    main(args)
