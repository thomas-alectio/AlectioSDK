import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import env
from model import Net
device = "cuda" if torch.cuda.is_available() else "cpu"

image_width, image_height = 32, 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def getdatasetstate():
    return {k:k for k in range(50000)}

def train(labeled, resume_from, ckpt_file):
    batch_size = 16
    lr = 1e-2
    weight_decay = 1e-2
    epochs = 10 # just for demo

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = Subset(trainset, labeled)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(env.EXPT_DIR, resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate()

    net.train()
    for epoch in tqdm(range(epochs), desc='Training'):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            images, labels = data
            images, labels = images.to(device), labels.type(torch.LongTensor).to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print('Finished Training. Saving the model as {}'.format(ckpt_file))
    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(env.EXPT_DIR, ckpt_file))

    return

def test(ckpt_file):
    batch_size = 16
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    predictions, targets = [], []
    net = Net().to(device)
    ckpt = torch.load(os.path.join(env.EXPT_DIR, ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Testing"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {"predictions": predictions, "labels": targets}

def infer(unlabeled, ckpt_file):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    unlabeled = Subset(trainset, unlabeled)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled, batch_size=4, shuffle=False, num_workers=2)

    net = Net().to(device)
    ckpt = torch.load(os.path.join(env.EXPT_DIR, ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total = 0, 0
	
    predictions = []
    with torch.no_grad():
        for data in tqdm(unlabeled_loader, desc="Inferring"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {"outputs": predictions}

if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)