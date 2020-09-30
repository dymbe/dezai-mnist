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
