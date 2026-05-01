from sklearn.metrics import mean_absolute_error


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred)
    }