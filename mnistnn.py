from model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


class MnistNN(Model):
    def __init__(self, device=torch.device("cuda"), state_path=None):
        self.device = device
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1))
        if state_path is not None:
            state_dict = torch.load(state_path)
            self.net.load_state_dict(state_dict)

    def train(self, train_loader, lr=1.0, epochs=5, verbose=False, plot=False, val_loader=None) -> None:
        net = self.net.to(self.device)
        optimizer = optim.SGD(self.net.parameters(), lr=lr)
        epoch_train_losses = []
        epoch_val_losses = []

        for epoch in range(epochs):
            if verbose:
                print("Epoch:", epoch)
            train_losses = []
            for (inputs, targets) in train_loader:
                net.train()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                loss = F.mse_loss(outputs, one_hot_targets)
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()
            epoch_train_losses.append(np.mean(train_losses))
            if val_loader is not None:
                epoch_val_losses.append(self.test_loss(val_loader))
            if plot:
                plt.plot(epoch_train_losses, c="blue")
                plt.plot(epoch_val_losses, c="red")
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
        losses = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = net(inputs)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                loss = F.mse_loss(output, one_hot_targets).item()
                losses.append(loss)
        return float(np.mean(losses))

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


def load_models(model_dir):
    dir_path = f"models/{model_dir}"
    models = []
    for file in os.listdir(dir_path):
        model = MnistNN(state_path=f"{dir_path}/{file}")
        models.append(model)
    return models
