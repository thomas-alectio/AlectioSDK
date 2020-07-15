# import os
# import torch
# import torchtext
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from torchtext.datasets import text_classification

# from tqdm import tqdm
# from model import TextSentiment

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from mushroom_data import MushroomDataset
from model import NeuralNet
import torch.optim as optim
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


def getdatasetstate(args={}):
    return {k: k for k in range(120000)}


def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    lr = 1e-4
    momentum = 0.9
    epochs = args["train_epochs"]

    global train_dataset, test_dataset

    CSV_FILE = "./data/datasets_478_974_mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    train_test = torch.utils.data.random_split(
        dataset, (int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset)))
    )

    train_dataset = train_test[0]
    test_dataset = train_test[1]

    # print("Train set length:", len(train))
    # print("Test set length:", len(test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    net = NeuralNet().to(device)

    print(net)

    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate()

    net.train()

    for epoch in tqdm(range(20), desc="Training"):

        running_loss = 0

        for i, batch in enumerate(train_loader, start=0):
            data, labels = batch
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000:
                print(
                    "epoch: {} batch: {} running-loss: {}".format(
                        epoch + 1, i + 1, running_loss / 1000
                    ),
                    end="\r",
                )
                running_loss = 0

    print("done training")

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    print("Training accuracy: {}".format((train_acc / len(chosen_train_dataset) * 100)))
    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

    return


def test(args, ckpt_file):
    batch_size = args["batch_size"]
    testloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=generate_batch
    )

    predictions, targets = [], []
    net = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader, desc="Testing"):
            text, offsets, cls = data
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            outputs = net(text, offsets)

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(cls.cpu().numpy().tolist())
            total += cls.size(0)
            correct += (predicted == cls).sum().item()

    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    unlabeled = Subset(train_dataset, unlabeled)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=generate_batch,
    )

    net = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    ckpt = torch.load(os.path.join(args["EXPT_DIR"], ckpt_file))
    net.load_state_dict(ckpt["model"])
    net.eval()

    correct, total, k = 0, 0, 0
    outputs_fin = {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(unlabeled_loader), desc="Inferring"):
            text, offsets, cls = data
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            outputs = net(text, offsets)

            _, predicted = torch.max(outputs.data, 1)
            total += cls.size(0)
            correct += (predicted == cls).sum().item()
            for j in range(len(outputs)):
                outputs_fin[k] = {}
                outputs_fin[k]["prediction"] = predicted[j].item()
                outputs_fin[k]["pre_softmax"] = outputs[j].cpu().numpy()
                k += 1

    return {"outputs": outputs_fin}


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
