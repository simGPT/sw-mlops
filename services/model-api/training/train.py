"""
로컬 학습용 (Docker 이미지에 포함되지 않음 -> 빌드 시간이 너무 오래걸려서 학습용은 따로 뺌)

의존성 설치:
pip install -r requirements-train.txt

conda 환경 활성화:
conda activate mlops

# v1 초기 학습
python3 training/train.py --version v1

# v2 재학습 (train + retrain 데이터 합쳐서 학습)
python3 training/train.py --version v2
"""
import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.mnist_model import get_model
from training.dataset import load_data, split_data
from training.trainer import train
from training.evaluate import evaluate


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # gpu 여부에 따라 없으면 cpu 사용
    print(f'Device: {device}')

    # 데이터 준비
    x, y, test_raw = load_data(args.data_dir) # 데이터 불러오는 함수
    x, y = split_data(x, y, test_raw) # 데이터셋 분리(학습 데이터, 재학습 데이터, 검증데이터, 테스트 데이터)
    x = [x_i.to(device) for x_i in x] # 데이터 이동 to gpu or to cpu
    y = [y_i.to(device) for y_i in y]

    # 학습 데이터 결정 -> v1: 첫학습, v2: 재학습
    if args.version == 'v1':
        x_train, y_train = x[0], y[0] # train만
    elif args.version == 'v2':
        x_train = torch.cat([x[0], x[1]], dim=0)  # train + retrain (인덱스1이 retrain 데이터)
        y_train = torch.cat([y[0], y[1]], dim=0)
    else:
        raise ValueError(f'잘못된 버전입니다: {args.version}')

    # 모델 가져오기
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # 최적화 Adam사용

    # 모델 학습
    model, _, _ = train(
        x_train, y_train,
        x[2], y[2],  # valid
        model, optimizer,
    )

    # 테스트
    evaluate(model, x[3], y[3])  # test

    # 모델 저장
    os.makedirs(args.artifact_dir, exist_ok=True)
    save_path = os.path.join(args.artifact_dir, f'model_{args.version}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'모델 저장 완료: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--artifact_dir', type=str, default='artifacts')
    args = parser.parse_args()

    main(args)
