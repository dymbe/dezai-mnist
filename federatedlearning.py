from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from numpy import ndarray
from mnistnn import Net
from utils import average_models, mv, wmv_cma, wmv, wmv_sma
from datasets import sorted_testset, uniform_mnist_subsets


def train_models(models, train_loaders, device, subepochs) -> list:
    models = [model.to(device) for model in models]
    optimizers = [optim.Adam(model.parameters()) for model in models]

    for i, model in enumerate(models):
        for epoch in range(subepochs):
            for (inputs, targets) in train_loaders[i]:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizers[i].zero_grad()
                output = model(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                loss = F.mse_loss(output, one_hot_targets)
                loss.backward()
                optimizers[i].step()
    return [model.cpu() for model in models]


def train_and_return(model, train_loader, device, subepochs, plot=True) -> Net:
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    losses = []
    for epoch in range(subepochs):
        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            targets2 = F.one_hot(targets, num_classes=10).float()
            loss = F.mse_loss(output, targets2)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    if plot:
        plt.plot(losses)
        plt.show()
    return model.cpu()


def test(model, device, test_loader) -> float:
    model = model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)


def all_outputs(models, device, test_loader) -> Tuple[ndarray, ndarray]:
    models = [model.to(device) for model in models]
    results = np.empty((len(test_loader.dataset), len(models), 10))
    all_targets = np.empty((len(test_loader.dataset), 1))

    with torch.no_grad():
        y = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            end = y + len(inputs)
            all_targets[y:end, 0] = targets.cpu()
            for x, model in enumerate(models):
                results[y:end, x] = model(inputs).cpu()
            y = end
    return results, all_targets


def experiment(batch_size,
               batches_per_client,
               trainset_size,
               testset_size,
               epochs,
               subepochs):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    dps_per_client = batch_size * batches_per_client

    n_clients = int(np.ceil(trainset_size / dps_per_client))

    train_subsets = uniform_mnist_subsets(n_clients, trainset_size)

    kwargs = {'batch_size': batch_size,
              'num_workers': 1,
              'pin_memory': True,
              'shuffle': False}
    train_loaders = [torch.utils.data.DataLoader(train_subset, **kwargs) for train_subset in train_subsets]

    x_test = sorted_testset(testset_size)
    test_loader = torch.utils.data.DataLoader(x_test, **kwargs)

    avg_model_scores = np.zeros(epochs)
    mv_scores = np.zeros(epochs)
    wmv_scores = np.zeros(epochs)
    wmv_sma_scores = np.zeros(epochs)
    wmv_cma_scores = np.zeros(epochs)
    model_scores = np.zeros((epochs, n_clients))

    models = [Net() for _ in range(n_clients)]

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # models = [copy.deepcopy(avg_model) for _ in range(n_clients)]
        models = train_models(models, train_loaders, device, subepochs)
        avg_model = average_models(models)

        avg_model_scores[epoch] = test(avg_model, device, test_loader)

        outputs, targets = all_outputs(models, device, test_loader)

        predictions = outputs.argmax(axis=2)

        majority_vote = mv(predictions)
        mv_scores[epoch] = np.mean(majority_vote == targets)

        b = 0.1
        wmv_scores[epoch] = np.mean(wmv(outputs, targets, b=b) == targets)

        n_sma = 50
        wmv_sma_scores[epoch] = np.mean(wmv_sma(outputs, targets, n=n_sma) == targets)

        wmv_cma_scores[epoch] = np.mean(wmv_cma(outputs, targets) == targets)

        model_scores[epoch] = np.mean(predictions == targets, axis=0)

        print("avg={:.2f}%, std={:.2f}%".format(100 * model_scores[epoch].mean(), 100 * model_scores[epoch].std()))
        print("Model performance:", ",\t".join(["{:.2f}%".format(100 * x) for x in model_scores[epoch]]))

        print("ehhh", 100 * wmv_sma_scores[epoch])

        x = range(epoch + 1)
        y_aggr = 100 * avg_model_scores[:epoch + 1]
        y_mv = 100 * mv_scores[:epoch + 1]
        y_wmv = 100 * wmv_scores[:epoch + 1]
        y_cma = 100 * wmv_cma_scores[:epoch + 1]
        y_sma = 100 * wmv_sma_scores[:epoch + 1]
        y_max = 100 * model_scores.max(axis=1)[:epoch + 1]
        y_min = 100 * model_scores.min(axis=1)[:epoch + 1]
        y_mean = 100 * model_scores.mean(axis=1)[:epoch + 1]
        stds = 100 * model_scores.std(axis=1)[:epoch + 1]

        #plt.plot(x, y_aggr, c="red", label="Aggregated (average) model")
        plt.plot(x, y_mv, c="purple", label="Majority vote")
        plt.plot(x, y_wmv, c="black", label=f"Weighted majority vote, b={b}")
        plt.plot(x, y_cma, c="teal", label="Weighted majority vote (CMA)")
        plt.plot(x, y_sma, c="green", label=f"Weighted majority vote (SMA), n={n_sma}")
        #plt.plot(x, y_max, c="green", label="Best individual model")
        #plt.plot(x, y_min, c="orange", label="Worst individual model")
        plt.errorbar(x, y_mean, yerr=stds, c="blue", label="Average of individual models")
        plt.legend()
        plt.title(f"|x_train|={trainset_size}, |x_test|={testset_size}, batch_size={batch_size}, batches_per_client={batches_per_client}")
        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("Score (%)")
        plt.show()


if __name__ == '__main__':
    experiment(batch_size=32,
               batches_per_client=1,
               trainset_size=1280,
               testset_size=10000,
               epochs=20,
               subepochs=1)
