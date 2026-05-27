import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


def train(
    x_train, y_train,   # 학습데이터
    x_valid, y_valid,   # 검증데이터
    model,  # 모델 함수
    optimizer,  # 최적화 알고리즘
    n_epochs: int = 1000,   # 에폭 수
    batch_size: int = 256,  # 배치 크기
    early_stop: int = 25,   # 조기 종료 조건
    print_interval: int = 10,   # 출력 간격
):
    device = next(model.parameters()).device
    crit = nn.NLLLoss()

    lowest_loss = np.inf # 무한대로 초기화 -> 그래야 작아질 수 있음
    lowest_epoch = np.inf
    best_model = None
    train_history, valid_history = [], []

    start_time = time.time()

    for i in range(n_epochs):
        # 학습
        indices = torch.randperm(x_train.size(0)).to(device)
        x_ = torch.index_select(x_train, dim=0, index=indices).split(batch_size, dim=0)
        y_ = torch.index_select(y_train, dim=0, index=indices).split(batch_size, dim=0)

        train_loss = 0
        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            loss = crit(y_hat_i, y_i.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss)
        train_loss /= len(x_) # 학습 손실값 평균

        # 검증
        with torch.no_grad():
            x_ = x_valid.split(batch_size, dim=0)
            y_ = y_valid.split(batch_size, dim=0)

            valid_loss = 0
            for x_i, y_i in zip(x_, y_):
                loss = crit(model(x_i), y_i.squeeze())
                valid_loss += float(loss)
            valid_loss /= len(x_)

        train_history.append(train_loss)
        valid_history.append(valid_loss)

        if (i + 1) % print_interval == 0:
            print(
                'Epoch %d: train_loss=%.4f  valid_loss=%.4e  lowest_loss=%.4e  elapsed=%.2fs'
                % (i + 1, train_loss, valid_loss, lowest_loss, time.time() - start_time)
            )

        if valid_loss <= lowest_loss:
            lowest_loss = valid_loss
            lowest_epoch = i
            best_model = deepcopy(model.state_dict())
        else:
            if early_stop > 0 and lowest_epoch + early_stop < i + 1: # 조기 종료
                print("%d 에폭 동안 모델 성능 향상되지 않아서 학습 종료." % early_stop)
                break

    print("최소 검증 손실 %d번째 에폭: %.4e" % (lowest_epoch + 1, lowest_loss))
    model.load_state_dict(best_model)

    return model, train_history, valid_history
