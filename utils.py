import numpy as np
from mnistnn import MnistNN


def mv(outputs):
    predictions = np.empty(len(outputs))
    for i, row in enumerate(outputs):
        counts = np.bincount(row.argmax(axis=1), minlength=10)
        predictions[i] = counts.argmax()
    return predictions


def average_models(models) -> MnistNN:
    model = MnistNN()
    sd = model.net.state_dict()
    for key in sd:
        sd[key] = sum([m.net.state_dict()[key] for m in models]) / len(models)
    model.net.load_state_dict(sd)
    return model


def load_and_average_models(model_loader) -> MnistNN:
    avg_model = MnistNN()
    avg_sd = avg_model.net.state_dict()
    for i, model in enumerate(model_loader):
        for key in avg_sd:
            sd = model.net.state_dict()
            avg_sd[key] = (sd[key] + i * avg_sd[key]) / (i + 1)
    avg_model.net.load_state_dict(avg_sd)
    return avg_model


def wmv_cma(outputs, targets):  # https://en.wikipedia.org/wiki/Moving_average#Cumulative_moving_average
    weights = np.ones(outputs.shape[1])
    predictions = np.empty(targets.shape)
    for y, target in enumerate(targets):
        if np.all(weights == 0):
            votes = np.sum(outputs[y], axis=0)
        else:
            votes = np.sum((outputs[y].T * np.square(weights)).T, axis=0)
        predictions[y] = votes.argmax()

        scores = outputs[y].argmax(axis=1) == target
        weights = (scores + y * weights) / (y + 1)  # Cumulative moving average score
    return predictions


def wmv_sma(outputs, targets, n=50):  # https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    weights = np.ones(outputs.shape[1])
    predictions = np.empty(targets.shape)
    all_scores = []

    for y, target in enumerate(targets):
        if np.all(weights == 0):
            votes = np.sum(outputs[y], axis=0)
        else:
            votes = np.sum((outputs[y].T * weights).T, axis=0)

        predictions[y] = votes.argmax()
        new_score = 1 * (outputs[y].argmax(axis=1) == target)

        if y < n:
            weights = (new_score + y * weights) / (y + 1)
        else:
            a = all_scores.pop(0)
            weights += (new_score - a) / n

        all_scores.append(new_score)

    return predictions


def wmv(outputs, targets, b):
    weights = np.ones(outputs.shape[1])
    predictions = np.empty(targets.shape)
    for y, target in enumerate(targets):
        count = np.zeros(outputs.shape[2])
        votes = outputs[y].argmax(axis=1)
        np.add.at(count, votes, weights)
        predictions[y] = count.argmax()
        weights *= 1 - b * (outputs[y].argmax(axis=1) != target)
    return predictions


def mv_wrong(outputs):
    votes = np.sum(outputs, axis=1)
    return np.argmax(votes, axis=1)
