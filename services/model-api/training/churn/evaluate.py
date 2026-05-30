from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 평가
def evaluate(model, x, y):
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]

    return {
        'accuracy': accuracy_score(y, pred),
        'f1': f1_score(y, pred),
        'roc_auc': roc_auc_score(y, proba),
    }
