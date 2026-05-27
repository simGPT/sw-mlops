import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix

# 테스트 데이터로 테스트
def evaluate(model, x_test, y_test, batch_size: int = 256):
    device = next(model.parameters()).device
    crit = nn.NLLLoss()

    x_ = x_test.split(batch_size, dim=0)
    y_ = y_test.split(batch_size, dim=0)

    test_loss = 0
    y_hat = []

    with torch.no_grad():
        for x_i, y_i in zip(x_, y_):
            y_hat_i = model(x_i)
            test_loss += float(crit(y_hat_i, y_i.squeeze()))
            y_hat.append(y_hat_i)

    test_loss /= len(x_)
    y_hat = torch.cat(y_hat, dim=0)

    preds = torch.argmax(y_hat, dim=1)
    accuracy = (y_test.squeeze() == preds).sum().item() / float(y_test.size(0))

    print('Test Loss: %.4e' % test_loss)
    print('Test Accuracy: %.4f' % accuracy)

    # 시각화
    cm = pd.DataFrame(
        confusion_matrix(y_test.cpu(), preds.cpu()),
        index=['true_%d' % i for i in range(10)],
        columns=['pred_%d' % i for i in range(10)],
    )
    print(cm)

    return test_loss, accuracy
