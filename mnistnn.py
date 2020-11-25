from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datasets import trainset, random_trainset, testset
import matplotlib.pyplot as plt
from copy import deepcopy


class MnistNN(Model):
    def __init__(self, device=torch.device("cuda")):
        self.device = device
        self.net = nn.Sequential(  # sequential operation
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1))

    def train(self, train_loader, epochs=10, verbose=False, plot=False) -> None:
        net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters())
        test_loss = []
        avg_epoch_loss = []

        for epoch in range(epochs):
            if verbose:
                print("Epoch:", epoch)
            losses = []
            for (inputs, targets) in train_loader:
                net.train()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                loss = F.mse_loss(outputs, one_hot_targets)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            if plot:
                avg_epoch_loss.append(np.mean(losses))
                print(my_net.test_score(ys))
                test_loss.append(self.test_loss(testset(10000)))
                plt.plot(test_loss, c="red")
                plt.plot(avg_epoch_loss, c="blue")
                plt.show()

        self.net = net.cpu()

    def test_score(self, test_loader) -> float:
        correct = 0
        net = self.net.to(self.device)
        net.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = net(inputs).argmax(dim=1)
                correct += torch.eq(predictions, targets).sum().item()
        return correct / len(test_loader.dataset)

    def test_loss(self, test_loader) -> float:
        net = self.net.to(self.device)
        net.eval()
        avg_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = net(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                loss = F.mse_loss(output, one_hot_targets).item()
                avg_loss = (loss + i * avg_loss) / (i + 1)  # Cumulative average
        return avg_loss

    def outputs(self, test_loader) -> np.ndarray:
        net = self.net.to(self.device)
        net.eval()
        outputs = np.empty((len(test_loader.dataset), 10))
        with torch.no_grad():
            y = 0
            for inputs, targets in test_loader:
                end = y + len(inputs)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs[y:end] = net(inputs).cpu()
                y = end
        return outputs

    def load(self, path):
        state_dict = torch.load(path)
        self.net.load_state_dict(state_dict)

    # def predict(self, inputs) -> np.ndarray:
    #     net = self.net.to(self.device)
    #     results = np.empty((len(test_loader.dataset), 1))
    #
    #     with torch.no_grad():
    #         y = 0
    #         for inputs, targets in test_loader:
    #             inputs, targets = inputs.to(self.device), targets.to(self.device)
    #             end = y + len(inputs)
    #             pred = net(inputs).argmax(dim=1)  # get the index of the max probability
    #             results[y:end, 0] = pred.cpu()
    #             y = end
    #     return results


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    xs = trainset(12000)
    ys = testset(10000)

    my_net = MnistNN()
    my_net.train(xs, epochs=5, verbose=True, plot=False)
    print(my_net.test_score(ys))
