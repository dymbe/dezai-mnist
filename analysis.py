import numpy as np
from datasets import testset
import mnistnn
from utils import mv, wmv, wmv_cma, load_and_average_models
from testrunner import get_results


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
    wmvs = []

    for i in range(10):
        project = f"m1875-ts60000-e5-lr1.0-v{i}"

        outputs, targets = get_results(project, 1875)

        if np.isnan(outputs).any():
            raise Exception("NaN-values found!")

        model_scores = scores(outputs, targets)

        model_loader = mnistnn.model_loader(project)
        fed_model = load_and_average_models(model_loader)
        fed_outputs = fed_model.outputs(xs)

        means.append(model_scores.mean())

        feds.append(score(fed_outputs, targets))
        mvs.append(mv_score(outputs, targets))
        wmvs.append(wmv_score(outputs, targets, b=0.1))

    print("mean mean", np.mean(means))
    print("mean std", np.std(means))

    print("feds mean", np.mean(feds))
    print("feds std", np.std(feds))

    print("mv mean", np.mean(mvs))
    print("mv std", np.std(mvs))

    print("wmv mean", np.mean(wmvs))
    print("wmv std", np.std(wmvs))
