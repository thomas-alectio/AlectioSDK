import os
import torch
import torchtext
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchtext.datasets import text_classification

from tqdm import tqdm
from model import TextSentiment

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def getdatasetstate(args={}):
    return {k: k for k in range(120000)}


def train(args, labeled, resume_from, ckpt_file):
    batch_size = args["batch_size"]
    lr = 4.0
    momentum = 0.9
    epochs = args["train_epochs"]

    global train_dataset, test_dataset
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./data', ngrams=args["N_GRAMS"], vocab=None)

    global VOCAB_SIZE, EMBED_DIM, NUN_CLASS
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = args["EMBED_DIM"]
    NUN_CLASS = len(train_dataset.get_labels())
    
    chosen_train_dataset = Subset(train_dataset, labeled)
    trainloader = DataLoader(chosen_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=generate_batch)    
    net = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    if resume_from is not None:
        ckpt = torch.load(os.path.join(args["EXPT_DIR"], resume_from))
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        getdatasetstate()

    net.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        running_loss = 0.0
        train_acc = 0
        for i, data in enumerate(trainloader):
            text, offsets, cls = data
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            outputs = net(text, offsets)
            loss = criterion(outputs, cls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += (outputs.argmax(1) == cls).sum().item()
            running_loss += loss.item()
        scheduler.step()

    print("Finished Training. Saving the model as {}".format(ckpt_file))
    print("Training accuracy: {}".format((train_acc / len(chosen_train_dataset) * 100)))
    ckpt = {"model": net.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(ckpt, os.path.join(args["EXPT_DIR"], ckpt_file))

    return


def test(args, ckpt_file):
    batch_size = args["batch_size"]
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=generate_batch) 

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
        unlabeled, batch_size=args["batch_size"], shuffle=False, num_workers=2, collate_fn=generate_batch)

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
