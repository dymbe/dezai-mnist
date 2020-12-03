import os
import numpy as np
from datasets import get_targets
import mnistnn
from datasets import testset


def test(model_dir, test_loader):
    out_dir_path = f"test_results/{model_dir}"

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    models = mnistnn.load_models(model_dir)

    outputs_path = f"test_results/{model_dir}/outputs.npy"
    targets_path = f"test_results/{model_dir}/targets.npy"

    if os.path.isfile(outputs_path):
        outputs = np.load(outputs_path)
    else:
        outputs = np.empty((len(test_loader.dataset), len(models), 10))
        outputs.fill(np.nan)

    targets = get_targets(test_loader)
    np.save(targets_path, targets)

    for i, model in enumerate(models):
        if not np.any(np.isnan(outputs[:, i])):
            print(f"Using cache", end="")
        else:
            outputs[:, i] = model.outputs(test_loader)
            np.save(outputs_path, outputs)
            print(f"Done testing", end="")
        print(f" - {i + 1}/{len(models)} models done")
    print("Done testing models!")

    return outputs, targets


if __name__ == '__main__':
    for i in range(10):
        test(f"m375-ts12000-e5-lr1.0-v{i}", testset())
