import torch
import os
import numpy as np
from datasets import random_trainsets
from mnistnn import MnistNN
import copy
import time
import datetime


def train_models(training_subsets, out_dir, epochs=5, lr=1.0, use_cache=True, init_model=None):
    out_dir = f"models/{out_dir}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    times = []
    for i, train_loader in enumerate(training_subsets):
        out_file = f"{out_dir}/model{i}.pt"
        if os.path.isfile(out_file) and use_cache:
            print("Model {:>4} found in cache - {:>4}/{} models trained".format(i, i + 1, len(training_subsets)))
            continue
        else:
            start_time = time.time()
            if init_model is None:
                model = MnistNN()
            else:
                model = copy.deepcopy(init_model)
            model.train(train_loader, epochs=epochs, lr=lr)
            torch.save(model.net.state_dict(), out_file)

            times.append(time.time() - start_time)
            avg_time = np.mean(times)
            time_left = np.round((len(training_subsets) - 1 - i) * avg_time)

            status = "Model {:>4} done training  - {:4}/{} models trained".format(i, i + 1, len(training_subsets))
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

    for i in range(10):
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
