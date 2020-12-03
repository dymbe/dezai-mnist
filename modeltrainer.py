import torch
import os
import numpy as np
from datasets import random_trainsets
from mnistnn import MnistNN
import copy
import time
import datetime


def train_models(training_subsets, out_dir, epochs=5, lr=1.0, use_cache=True, init_model=None):
    out_dir_path = f"models/{out_dir}"
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    times = []
    for i, train_loader in enumerate(training_subsets):
        start_time = time.time()

        path = f"{out_dir_path}/model{i}.pt"
        if os.path.isfile(path) and use_cache:
            print(f"Model {i} found in cache - {i + 1}/{len(training_subsets)} models trained")
            continue
        else:
            if init_model is None:
                model = MnistNN()
            else:
                model = copy.deepcopy(init_model)
            model.train(train_loader, epochs=epochs, lr=lr)
            torch.save(model.net.state_dict(), path)

            times.append(time.time() - start_time)
            avg_time = np.mean(times)
            time_left = np.round((len(training_subsets) - i) * avg_time)

            status = f"Model {i} done training  - {i + 1}/{len(training_subsets)} models trained"
            status += f" (avg. time: {'{:.2f}s, time left: {}'.format(avg_time, datetime.timedelta(seconds=time_left))}"
            print(status)

    print("Done!")


if __name__ == '__main__':
    models = 1875
    train_size = 60000
    my_epochs = 5
    my_lr = 1.0
    same_init = False

    torch.manual_seed(0)
    np.random.seed(0)

    for i in range(1):
        my_out_dir = f"m{models}-ts{train_size}-e{my_epochs}-lr{my_lr}"
        if same_init:
            my_out_dir += '-init'
            my_init_model = MnistNN()
        else:
            my_init_model = None
        my_out_dir += f"-v{i}"
        print(my_out_dir)
        train_models(random_trainsets(models, train_size), my_out_dir,
                     epochs=my_epochs, lr=my_lr, init_model=my_init_model)
