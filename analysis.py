import numpy as np
from datasets import testset
import mnistnn
from utils import mv, wmv, wmv_cma, average_models


def score(output, targets):
    return np.mean(output.argmax(axis=1) == targets)


def mv_score(outputs, targets):
    return np.mean(mv(outputs) == targets)


def wmv_score(outputs, targets, b):
    return np.mean(wmv(outputs, targets, b) == targets)


def cma_score(outputs, targets):
    return np.mean(wmv_cma(outputs, targets) == targets)


def scores(output, targets):
    return np.mean(output.argmax(axis=2) == targets.reshape(-1, 1), axis=0)


if __name__ == '__main__':
    xs = testset()

    means = []
    feds = []
    mvs = []

    for i in range(10):
        project = f"m375-ts12000-e5-lr1.0-init-v{i}"

        outputs = np.load(f"test_results/{project}/outputs.npy")

        if np.isnan(outputs).any():
            raise Exception("NaN-values found!")

        targets = np.load(f"test_results/{project}/targets.npy")

        model_scores = scores(outputs, targets)

        models = mnistnn.load_models(project)
        fed_model = average_models(models)
        fed_outputs = fed_model.outputs(xs)

        means.append(model_scores.mean())

        feds.append(score(fed_outputs, targets))
        mvs.append(mv_score(outputs, targets))

    print("mean mean", np.mean(means))
    print("mean std", np.std(means))

    print("feds mean", np.mean(feds))
    print("feds std", np.std(feds))

    print("mv mean", np.mean(mvs))
    print("mv std", np.std(mvs))
