import torch
from torchvision import datasets, transforms

# 데이터 불러오는 함수
def load_data(data_dir: str):
    train_raw = datasets.MNIST(
        data_dir, train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_raw = datasets.MNIST(
        data_dir, train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    x = train_raw.data.float() / 255.
    y = train_raw.targets
    x = x.view(x.size(0), -1)

    return x, y, test_raw

# 데이터셋 분리하는 함수
def split_data(x, y, test_raw, ratio=(0.5, 0.3, 0.2)):
    """
    데이터셋 내부
        x[0], y[0] : train      (초기 학습)
        x[1], y[1] : retrain    (재학습용 데이터)
        x[2], y[2] : valid      (검증 데이터)
        x[3], y[3] : test       (테스트 데이터)
    """
    train_cnt   = int(x.size(0) * ratio[0])
    retrain_cnt = int(x.size(0) * ratio[1])
    valid_cnt   = x.size(0) - train_cnt - retrain_cnt
    test_cnt    = len(test_raw.data)

    print("Train %d / Retrain %d / Valid %d / Test %d samples."
          % (train_cnt, retrain_cnt, valid_cnt, test_cnt))

    indices = torch.randperm(x.size(0))
    x = torch.index_select(x, dim=0, index=indices)
    y = torch.index_select(y, dim=0, index=indices)

    x = list(x.split([train_cnt, retrain_cnt, valid_cnt], dim=0))
    y = list(y.split([train_cnt, retrain_cnt, valid_cnt], dim=0))

    x += [(test_raw.data.float() / 255.).view(test_cnt, -1)]
    y += [test_raw.targets]

    return x, y
