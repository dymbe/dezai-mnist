import numpy as np
import copy


def mv(outputs):
    predictions = np.empty(len(outputs))
    for i, row in enumerate(outputs):
        counts = np.bincount(row.argmax(axis=1), minlength=10)
        predictions[i] = counts.argmax()
    return predictions


def average_models(models):
    model = copy.deepcopy(models[0])
    sd = model.state_dict()
    for key in sd:
        sd[key] = sum([model.state_dict()[key] for model in models]) / len(models)
    model.load_state_dict(sd)
    return model


def wmv_cma(outputs, targets):  # https://en.wikipedia.org/wiki/Moving_average#Cumulative_moving_average
    weights = np.ones(outputs.shape[1])
    predictions = np.empty(targets.shape)
    for y, _ in enumerate(targets):
        if np.all(weights == 0):
            votes = np.sum(outputs[y], axis=0)
        else:
            votes = np.sum((outputs[y].T * weights).T, axis=0)
        predictions[y, 0] = votes.argmax()

        scores = outputs[y].argmax(axis=1) == targets[y]
        weights = (scores + y * weights) / (y + 1)  # Cumulative moving average score
    return predictions


def wmv_sma(outputs, targets, n=50):  # https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
    weights = np.ones(outputs.shape[1])
    predictions = np.empty(targets.shape)
    all_scores = []

    for y, _ in enumerate(targets):
        if np.all(weights == 0):
            votes = np.sum(outputs[y], axis=0)
        else:
            votes = np.sum((outputs[y].T * weights).T, axis=0)

        predictions[y] = votes.argmax()
        #print("\t".join(["{:.2f}".format(x) for x in list(votes)]), "->", int(predictions[y][0]), ":", int(targets[y][0]))

        new_score = 1 * (outputs[y].argmax(axis=1) == targets[y])

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
    for y, _ in enumerate(targets):
        votes = np.sum((outputs[y].T * weights).T, axis=0)
        predictions[y] = votes.argmax()
        weights *= 1 - b * (outputs[y].argmax(axis=1) != targets[y])
    return predictions


def mv_wrong(outputs):
    votes = np.sum(outputs, axis=1)
    return np.argmax(votes, axis=1)
