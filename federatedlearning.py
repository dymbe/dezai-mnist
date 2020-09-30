from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import copy
from torchvision import datasets, transforms
from mnistnn import Net
from utils import average_models, average_optimizers, most_frequent_in_rows


def train_models(models, train_loaders, device, subepochs, optimizer_params=None):
    models = [model.to(device) for model in models]
    optimizers = [optim.Adam(model.parameters()) for model in models]
    if optimizer_params is not None:
        for optimizer in optimizers:
            optimizer.param_groups[0]["params"] = optimizer_params

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

    avg_params = average_optimizers(optimizers)

    return [model.cpu() for model in models], avg_params


def train_and_return(model, train_loader, device, subepochs, plot=True):
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


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
    print('Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset),
                                             100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def all_outputs(models, device, test_loader):
    models = [model.to(device) for model in models]
    results = np.empty((len(test_loader.dataset), len(models)))
    all_targets = np.empty((len(test_loader.dataset), 1))

    with torch.no_grad():
        y = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            end = y + len(inputs)
            all_targets[y:end, 0] = targets.cpu()
            for x, model in enumerate(models):
                output = model(inputs)
                pred = output.argmax(dim=1)  # get the index of the max probability
                results[y:end, x] = pred.cpu()
            y = end
    return results, all_targets


def experiment(batch_size,
               batches_per_client,
               dataset_size,
               testset_size,
               epochs,
               subepochs):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dps_per_client = batch_size * batches_per_client

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    x_train = datasets.MNIST('data', train=True, download=True,
                             transform=transform)

    n_clients = int(np.ceil(dataset_size / dps_per_client))

    train_subset_sizes = [dps_per_client] * n_clients
    if dataset_size % dps_per_client != 0:
        train_subset_sizes[n_clients - 1] = dataset_size % dps_per_client

    if dataset_size < len(x_train):
        train_subset_sizes += [len(x_train) - dataset_size]

    train_subsets = torch.utils.data.random_split(x_train, train_subset_sizes, None)
    kwargs = {'batch_size': batch_size,
              'num_workers': 1,
              'pin_memory': True,
              'shuffle': True}
    train_loaders = [torch.utils.data.DataLoader(train_subset, **kwargs) for train_subset in train_subsets]

    x_test = datasets.MNIST('data', train=False,
                            transform=transform)
    x_test = torch.utils.data.random_split(x_test, [testset_size, len(x_test) - testset_size], None)[0]
    test_loader = torch.utils.data.DataLoader(x_test, **kwargs)

    avg_model_scores = np.zeros(epochs)
    majority_vote_scores = np.zeros(epochs)
    model_scores = np.zeros((epochs, n_clients))

    avg_model = Net()
    optimizer_params = None

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        models = [copy.deepcopy(avg_model) for _ in range(n_clients)]
        models, optimizer_params = train_models(models, train_loaders, device, subepochs, optimizer_params=optimizer_params)
        avg_model = average_models(models)

        avg_model_scores[epoch] = test(avg_model, device, test_loader)

        outputs, targets = all_outputs(models, device, test_loader)

        majority_vote = most_frequent_in_rows(outputs)
        majority_vote_scores[epoch] = 100 * np.mean(majority_vote == targets)

        model_scores[epoch, :] = 100 * np.mean(outputs == targets, axis=0)

        x = range(epoch + 1)
        y1 = avg_model_scores[:epoch + 1]
        y2 = majority_vote_scores[:epoch + 1]
        y3 = model_scores.max(axis=1)[:epoch + 1]
        y4 = model_scores.min(axis=1)[:epoch + 1]
        y5 = model_scores.mean(axis=1)[:epoch + 1]
        stds = model_scores.std(axis=1)[:epoch + 1]

        plt.plot(x, y1, c="red", label="Aggregated (average) model")
        plt.plot(x, y2, c="purple", label="Majority vote")
        plt.plot(x, y3, c="green", label="Best individual model")
        plt.plot(x, y4, c="orange", label="Worst individual model")
        plt.errorbar(x, y5, yerr=stds, c="blue", label="Average of individual models")
        plt.legend()
        plt.title(f"|x_train|={dataset_size}, |x_test|={testset_size}, batch_size={batch_size}, batches_per_client={batches_per_client}")
        plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("Score (%)")
        plt.show()


if __name__ == '__main__':
    experiment(batch_size=32,
               batches_per_client=2,
               dataset_size=12800,
               testset_size=1000,
               epochs=20,
               subepochs=1)
