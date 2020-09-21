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


def train_and_return(model, train_loader, device, subepochs, plot=True):
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    losses = []
    for epoch in range(subepochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
    model.eval()
    model = model.to(device)
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    print('Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset),
                                             100. * correct / len(test_loader.dataset)))


#get_results


def average_models(models):
    model = Net()
    sd = model.state_dict()
    for key in sd:
        sd[key] = sum([model.state_dict()[key] for model in models]) / len(models)
    model.load_state_dict(sd)
    return model


def experiment(batch_size,
               batches_per_client,
               epochs,
               subepochs,
               num_tests,
               plot):

    torch.manual_seed(0)
    device = torch.device("cuda")
    dps_per_client = batch_size * batches_per_client

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    x_train = datasets.MNIST('data', train=True, download=True,
                             transform=transform)

    n_clients = int(np.ceil(len(x_train) / dps_per_client))

    subset_sizes = [dps_per_client] * n_clients
    if len(x_train) % dps_per_client != 0:
        subset_sizes[n_clients - 1] = len(x_train) % dps_per_client

    subsets = torch.utils.data.random_split(x_train, subset_sizes, generator=torch.Generator().manual_seed(42))

    kwargs = {'batch_size': batch_size,
              'num_workers': 1,
              'pin_memory': True,
              'shuffle': True}

    avg_model = Net()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        models = [copy.deepcopy(avg_model) for _ in range(n_clients)]

        for i in range(n_clients):
            print(f"Training model {i + 1}...")
            train_loader = torch.utils.data.DataLoader(subsets[i], **kwargs)
            models[i] = train_and_return(models[i], train_loader, device, subepochs, plot=plot)

        avg_model = average_models(models)

        x_test = datasets.MNIST('data', train=False,
                                transform=transform)
        test_loader = torch.utils.data.DataLoader(x_test, **kwargs)

        print("\nAverage model:")
        test(avg_model, device, test_loader)

        print("\nNormal models:")
        for model in models[:num_tests]:
            test(model, device, test_loader)


if __name__ == '__main__':
    experiment(batch_size=32,
               batches_per_client=1,
               epochs=20,
               subepochs=1,
               num_tests=5,
               plot=False)
