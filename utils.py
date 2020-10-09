import numpy as np
import copy
import torch


def most_frequent_in_rows(a):
    out = np.empty((a.shape[0], 1))
    for i, row in enumerate(a):
        out[i, 0] = np.bincount(row.astype(int)).argmax()
    return out


def average_models(models):
    model = copy.deepcopy(models[0])
    sd = model.state_dict()
    for key in sd:
        sd[key] = sum([model.state_dict()[key] for model in models]) / len(models)
    model.load_state_dict(sd)
    return model


def average_optimizers(optimizers):
    optimizer_params = [optimizer.param_groups[0]["params"] for optimizer in optimizers]
    mean_params = []
    for i in range(len(optimizer_params[0])):
        layers_i = torch.stack([p[i] for p in optimizer_params])
        mean = torch.mean(layers_i, dim=0)
        mean_params.append(mean)
    return mean_params


def wmv(outputs, targets):
    num_models = outputs.shape[1]
    weights = np.ones(num_models)
    predictions = np.empty(targets.shape)
    for y, target in enumerate(targets):
        votes = np.zeros(outputs.shape[2])
        for x, _ in enumerate(outputs[y]):
            votes += weights[x] * outputs[y, x]
            weights[x] = np.mean(outputs[:y + 1, x].argmax(axis=1) == targets[:y + 1, 0])
        predictions[y, 0] = votes.argmax()
    return predictions


def wmv_real(outputs, targets, b):
    num_models = outputs.shape[1]
    weights = np.ones(num_models)
    predictions = np.empty(targets.shape)
    for y, _ in enumerate(targets):
        score = np.sum((outputs[y].T * weights).T, axis=0)
        predictions[y, 0] = score.argmax()
        weights *= 1 - b * (outputs[y].argmax(axis=1) != targets[y, 0])
    return predictions
