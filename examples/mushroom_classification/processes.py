from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from mushroom_data import MushroomDataset
from model import NeuralNet
import torch.optim as optim
import os
import yaml
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def getdatasetstate(config_args={}):  
    return {k: k for k in range(config_args["train_size"])}

def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    
    CSV_FILE = "./data/datasets_478_974_mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    train_test = torch.utils.data.random_split(dataset, (int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))))

    train_loader = DataLoader(train_test[0], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_test[1], batch_size=batch_size, shuffle=True)
    
    net = NeuralNet()
    net = net.to(device=device)

    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate(args)

    net.train()

    for epoch in tqdm(range(args["train_epochs"]), desc="Training"):

        running_loss = 0
    
        for i, batch in enumerate(train_loader, start=0):
            data, labels = batch
            
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i%1000):
                print("epoch: {} batch: {} running-loss: {}".format(epoch + 1, i + 1, running_loss/1000), end="\r")
                running_loss = 0

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))
    return


def test(args, ckpt_file):
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]
    
    CSV_FILE = "./data/datasets_478_974_mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    train_test = torch.utils.data.random_split(dataset, (int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))))

    train_loader = DataLoader(train_test[0], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_test[1], batch_size=batch_size, shuffle=True)
    
    net = NeuralNet()
    net = net.to(device=device)

    net.load_state_dict(torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"])

    net.eval()
    predix = 0
    predictions = {}
    truelabels = {}
    
    n_val = len(train_test[0])
    with tqdm(total=n_val, desc='Testing round', unit='batch', leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(test_loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                prediction = net(batch_x)
                
            for logit,label  in zip(prediction,batch_y):
                predictions[predix] = logit.cpu().numpy().tolist()
                truelabels[predix] = label.cpu().numpy().tolist()
                predix+=1
            
            pbar.update()
            
    return {"predictions": predictions, "labels": truelabels}


def infer(args, unlabeled, ckpt_file):
    batch_size = args["batch_size"]
    lr = args["learning_rate"]
    momentum = args["momentum"]
    epochs = args["train_epochs"]

    global train_dataset, test_dataset
    
    CSV_FILE = "./data/datasets_478_974_mushrooms.csv"
    dataset = MushroomDataset(CSV_FILE)

    train_test = torch.utils.data.random_split(dataset, (int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))))

    train_loader = DataLoader(train_test[0], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_test[1], batch_size=batch_size, shuffle=True)

    train_loader = Subset(train_dataset, unlabeled)
    
    net = NeuralNet()
    net = net.to(device=device)

    print(net)
    net.load_state_dict(torch.load(os.path.join(args["EXPT_DIR"], ckpt_file + ".pth"))["model"])

    net.eval()
    
    n_val = len(train_dataset)
    predictions = {}
    predix = 0
    with tqdm(total=n_val, desc='Inference round', unit='batch', leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            with torch.no_grad():
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                prediction = net(batch_x)
                
            for logit in prediction:
                predictions[predix] = logit.cpu().numpy().tolist()
                predix+=1
            
            pbar.update()
    
    return {"outputs": predictions}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.getcwd(), "config.yaml"),
        type=str,
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        args = yaml.safe_load(stream)

    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"


    print("Testing getdatastate")
    getdatasetstate(config_args=args)
    print("Running train")
    train(args=args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    print("Running test")
    test(args=args, ckpt_file=ckpt_file)
    print("Running infer")
    infer(args=args, unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
