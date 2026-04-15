import torch.nn as nn

INPUT_SIZE = 784   # 28 x 28
OUTPUT_SIZE = 10   # 0 ~ 9


def get_model():
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, 500),
        nn.LeakyReLU(),
        nn.Linear(500, 400),
        nn.LeakyReLU(),
        nn.Linear(400, 300),
        nn.LeakyReLU(),
        nn.Linear(300, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 100),
        nn.LeakyReLU(),
        nn.Linear(100, 50),
        nn.LeakyReLU(),
        nn.Linear(50, OUTPUT_SIZE),
        nn.LogSoftmax(dim=-1),
    )
