import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import copy
from torchvision import datasets, transforms
from mnistnn import Net
from utils import average_models, most_frequent_in_rows, wmv, wmv_real


def train_models(models, train_loaders, device, subepochs):
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
    return correct / len(test_loader.dataset)


def all_outputs(models, device, test_loader):
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
    mv_scores = np.zeros(epochs)
    wmv_scores = np.zeros(epochs)
    wmv_real_scores = np.zeros(epochs)
    wmv_real_scores2 = np.zeros(epochs)
    wmv_real_scores3 = np.zeros(epochs)
    wmv_real_scores4 = np.zeros(epochs)
    model_scores = np.zeros((epochs, n_clients))

    avg_model = Net()
    models = [copy.deepcopy(avg_model) for _ in range(n_clients)]

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # models = [copy.deepcopy(avg_model) for _ in range(n_clients)]
        models = train_models(models, train_loaders, device, subepochs)
        avg_model = average_models(models)

        avg_model_scores[epoch] = test(avg_model, device, test_loader)

        outputs, targets = all_outputs(models, device, test_loader)
        predictions = outputs.argmax(axis=2)

        majority_vote = most_frequent_in_rows(predictions)
        mv_scores[epoch] = np.mean(majority_vote == targets)
        wmv_scores[epoch] = np.mean(wmv(outputs, targets) == targets)

        bs = [0.05, 0.2, 0.5, 0.9]
        wmv_real_scores[epoch] = np.mean(wmv_real(outputs, targets, b=bs[0]) == targets)
        wmv_real_scores2[epoch] = np.mean(wmv_real(outputs, targets, b=bs[1]) == targets)
        wmv_real_scores3[epoch] = np.mean(wmv_real(outputs, targets, b=bs[2]) == targets)
        wmv_real_scores4[epoch] = np.mean(wmv_real(outputs, targets, b=bs[3]) == targets)

        model_scores[epoch, :] = np.mean(predictions == targets, axis=0)

        print("avg={:.2f}, std={:.2f}".format(100 * model_scores[epoch].mean(), 100 * model_scores[epoch].std()))

        x = range(epoch + 1)
        y1 = 100 * avg_model_scores[:epoch + 1]
        y2 = 100 * mv_scores[:epoch + 1]
        y3 = 100 * wmv_scores[:epoch + 1]
        y4 = 100 * wmv_real_scores[:epoch + 1]
        y4_2 = 100 * wmv_real_scores[:epoch + 1]
        y4_3 = 100 * wmv_real_scores[:epoch + 1]
        y4_4 = 100 * wmv_real_scores[:epoch + 1]
        y5 = 100 * model_scores.max(axis=1)[:epoch + 1]
        y6 = 100 * model_scores.min(axis=1)[:epoch + 1]
        y7 = 100 * model_scores.mean(axis=1)[:epoch + 1]
        stds = 100 * model_scores.std(axis=1)[:epoch + 1]

        # plt.plot(x, y1, c="red", label="Aggregated (average) model")
        plt.plot(x, y2, c="purple", label="Majority vote")
        plt.plot(x, y3, c="teal", label="Weighted majority vote")
        plt.plot(x, y4, c="black", label="Weighted majority vote 2, b={}".format(bs[0]))
        plt.plot(x, y4_2, c="gray", label="Weighted majority vote 2, b={}".format(bs[1]))
        plt.plot(x, y4_3, c="green", label="Weighted majority vote 2, b={}".format(bs[2]))
        plt.plot(x, y4_4, c="orange", label="Weighted majority vote 2, b={}".format(bs[3]))
        # plt.plot(x, y5, c="green", label="Best individual model")
        # plt.plot(x, y6, c="orange", label="Worst individual model")
        plt.errorbar(x, y7, yerr=stds, c="blue", label="Average of individual models")
        plt.legend()
        plt.title(f"|x_train|={dataset_size}, |x_test|={testset_size}, batch_size={batch_size}, batches_per_client={batches_per_client}")
        #plt.ylim(0, 100)
        plt.xlabel("Epoch")
        plt.ylabel("Score (%)")
        plt.show()


if __name__ == '__main__':
    experiment(batch_size=32,
               batches_per_client=2,
               dataset_size=640,
               testset_size=100,
               epochs=5,
               subepochs=10)
