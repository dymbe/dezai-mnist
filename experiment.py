import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import random_trainsets, sorted_testset, testset, random_subsets
from mnistnn import MnistNN
from testrunner import load_models
from utils import mv, wmv, wmv_sma, wmv_cma, mv_wrong


def outputs_gpu_models(models, device, test_loader):
    outputs = np.empty((len(test_loader.dataset), len(models), 10))
    models = [model.net.to(device) for model in models]
    all_targets = np.empty((len(test_loader.dataset)))

    y = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        end = y + len(inputs)
        all_targets[y:end] = targets.cpu()
        for x, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                outputs[y:end, x] = model(inputs).cpu()
        y = end
    return outputs, all_targets


def experiment(models, device, test_loader, uniform_test=True):
    outputs, targets = outputs_gpu_models(models, device, test_loader)
    predictions = outputs.argmax(axis=2)

    x = np.linspace(1, outputs.shape[0], num=outputs.shape[0])

    b = 0.1
    n_sma = 50

    mv_scores = np.cumsum(mv_wrong(outputs) == targets.reshape(-1), axis=0) / x.reshape(-1)
    wmv_scores = np.cumsum(wmv(outputs, targets, b=b) == targets, axis=0) / x
    wmv_sma_scores = np.cumsum(wmv_sma(outputs, targets, n=n_sma) == targets, axis=0) / x
    #wmv_cma_scores = np.cumsum(wmv_cma(outputs, targets) == targets, axis=0) / x
    #all_model_scores = np.cumsum(predictions == targets.reshape(-1, 1), axis=0) / x
    #mean_model_scores = np.mean(all_model_scores, axis=1)
    #std_model_scores = np.std(all_model_scores, axis=1)

    if not uniform_test:
        label_borders = [0] + [indices.max() for indices in [np.where(targets == i)[0] for i in range(10)]]
        for i in range(0, len(label_borders) - 1, 2):
            plt.fill_between(x, 0, 100, where=np.logical_and(label_borders[i] <= x, x < label_borders[i + 1]), color='grey',
                             alpha=0.5)

    # for i, b in enumerate(np.linspace(0, 1, 50)):
    #     wmv_scores = np.cumsum(wmv(outputs, targets, b=b) == targets, axis=0).reshape(-1) / x
    #     plt.plot(x, 100 * wmv_scores, c="black", label=f"Weighted majority vote, b={b}")

    plt.plot(x, 100 * mv_scores, c="pink", label=f"Majority vote")
    plt.plot(x, 100 * wmv_scores, c="black", label=f"Weighted majority vote, b={b}")
    plt.plot(x, 100 * wmv_sma_scores, c="green", label=f"Weighted majority vote (SMA), n={n_sma}")
    #plt.plot(x, 100 * wmv_cma_scores, c="orange", label="Weighted majority vote (CMA)")
    #plt.errorbar(x, 100 * mean_model_scores, yerr=(100 * std_model_scores), ecolor="lightblue", c="blue",
    #             label="Average of individual models")

    plt.legend()
    #plt.title(f"|x_train|={trainset_size}, |x_test|={testset_size}, batch_size={batch_size}, batches_per_client={batches_per_client}")
    plt.ylim(0, 100)
    plt.xlabel("Test-examples")
    plt.ylabel("Score (%)")
    plt.show()

    labelwise_scores = []
    for i in range(10):
        indices = np.where(targets == i)[0]
        labelwise_scores.append(np.mean(predictions[indices, :] == targets[indices]))

    print(labelwise_scores)
    counts = np.bincount(predictions.reshape(-1))
    print(list(counts / counts.sum()))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    test_loader = testset(10000)

    models = load_models("m375-ts12000-e5")

    experiment(models, device, test_loader)


if __name__ == '__main__':
    main()
