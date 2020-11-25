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
    def __init__(self, device=torch.device("cuda"), state_path=None):
        self.device = device
        self.net = nn.Sequential(  # sequential operation
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1))
        if state_path is not None:
            state_dict = torch.load(state_path)
            self.net.load_state_dict(state_dict)

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
