import os
import numpy as np
from datasets import get_targets
import mnistnn
from datasets import testset
import time
import datetime
import torch


def test(project_name, test_loader, num_models, use_cache=True):
    out_dir = f"test_results/{project_name}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    targets_path = f"test_results/{project_name}/targets.npy"
    targets = get_targets(test_loader)
    np.save(targets_path, targets)

    times = []

    for i, model in enumerate(mnistnn.model_loader(project_name)):
        out_file = f"{out_dir}/model{i}.npy"
        if os.path.isfile(out_file) and use_cache:
            print(f"Model {i} found in cache - {i + 1}/{num_models} models trained")
            continue
        else:
            start_time = time.time()

            out = model.outputs(test_loader)
            np.save(out_file, out)

            times.append(time.time() - start_time)
            avg_time = np.mean(times)
            time_left = np.round((num_models - 1 - i) * avg_time)

            status = f"Model {i} tested - {i + 1}/{num_models} models tested"
            status += f" (avg. time: {'{:.2f}s, time left: {}'.format(avg_time, datetime.timedelta(seconds=time_left))}"
            print(status)

    print("Done testing models!")


def get_results(project_name, num_models):
    result_dir = f"test_results/{project_name}"
    example_result = np.load(f"{result_dir}/model0.npy")
    outputs_shape = (example_result.shape[0], num_models, example_result.shape[1])
    outputs = np.zeros(outputs_shape)

    for i in range(num_models):
        result_file = f"{result_dir}/model{i}.npy"
        model_result = np.load(result_file)
        outputs[:, i, :] = model_result

    target_file = f"{result_dir}/targets.npy"
    targets = np.load(target_file)

    return outputs, targets


if __name__ == '__main__':
    num_models = 1875
    train_size = 60000
    epochs = 5
    lr = 1.0
    same_init = False

    torch.manual_seed(0)
    np.random.seed(0)

    project_name = f"m{num_models}-ts{train_size}-e{epochs}-lr{lr}"
    if same_init:
        project_name += '-init'

    for i in range(1):
        project_name += f"-v{i}"
        test(project_name, testset(), num_models)
