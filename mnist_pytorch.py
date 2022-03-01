#!/usr/local/bin/python3
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST


def getMNISTDataLoader(batch_size):
    print("Preparing data ...")
    train_subset_indices = np.arange(1000)
    test_subset_indices = np.arange(100)
    transform = transforms.ToTensor()
    trainset = MNIST(root="./data", train=True, download=True, transform=transform)
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    sub_trainset = Subset(trainset, train_subset_indices)
    sub_testset = Subset(testset, test_subset_indices)
    trainloader = DataLoader(
        sub_trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        sub_testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    print("Done")
    return trainloader, testloader


def train(model, dataloader, optimizer):
    model.train()
    total_loss = []
    for data in dataloader:
        x, y = data
        _y = model(x)
        loss = model.calc_loss(_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    return np.mean(total_loss)


def evaluate(model, dataloader):
    model.eval()
    total_loss = []
    total_pred = []
    total_y = []
    with torch.no_grad():
        for data in dataloader:
            x, y = data
            _y = model(x)
            loss = model.calc_loss(_y, y)
            total_loss.append(loss.item())
            total_pred += torch.argmax(
                torch.softmax(_y.detach().cpu(), -1), -1
            ).tolist()
            total_y += y.tolist()

    result = {
        "loss": np.mean(total_loss),
        "acc": accuracy_score(total_y, total_pred),
    }
    return result


def print_result(state, result):
    print(f">> {state} <<")
    for k, v in result.items():
        print(f". {k}: {v:.4f}")


def run(model, trainloader, testloader, optimizer, n_epochs):
    print("Start Training ...")
    for epoch in range(1, n_epochs + 1):
        avg_loss = train(model, trainloader, optimizer)
        print(f"\nEpoch {epoch}/{n_epochs} avg_loss: {avg_loss:.4f}")
        train_result = evaluate(model, trainloader)
        print_result("train", train_result)
        test_result = evaluate(model, testloader)
        print_result("test", test_result)
    print("\nFinished Training")


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.to(self.device).float()
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def calc_loss(self, _y, y):
        return self.loss_fn(_y, y.to(self.device))


if __name__ == "__main__":
    batch_size = 128
    n_epochs = 10
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = getMNISTDataLoader(batch_size)
    model = Net(device).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    run(model, trainloader, testloader, optimizer, n_epochs)
