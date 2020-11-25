import torch
import os
import numpy as np
from datasets import random_trainsets
from mnistnn import MnistNN


def train_models(training_subsets, out_dir, epochs=5, use_cache=True):
    out_dir_path = f"models/{out_dir}"
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    for i, train_loader in enumerate(training_subsets):
        path = f"{out_dir_path}/model{i}.pt"
        if os.path.isfile(path) and use_cache:
            print(f"Model {i}: found in cache")
            continue
        else:
            model = MnistNN()
            model.train(train_loader, epochs=epochs)
            torch.save(model.net.state_dict(), path)
            print(f"Model {i}: trained")

    print("Done!")


if __name__ == '__main__':
    models = 1
    train_size = 12000
    epochs = 5

    torch.manual_seed(0)
    np.random.seed(0)
    train_models(random_trainsets(models, train_size), f"m{models}-ts{train_size}-e{epochs}", epochs=epochs)
