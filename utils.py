import torch
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, \
    CosineAnnealingLR, LinearLR, ExponentialLR

from config import Config


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def initialise_optimisers(config, model):
    learning_rate = 10 ** (-config.train_config["learning_rate_exp"])
    optimiser = config.train_config["optimiser"]

    if optimiser == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimiser == "AdamW":
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimiser == "SGD":
        optimiser = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001
        )

    return optimiser

def initialise_schedular(config, optimizer):
    scheduler = config.train_config["schedular"]

    if scheduler == "ReduceOnPlateau":
       scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    elif scheduler == "Cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max = config.train_config["epochs"])
    elif scheduler == "Linear":
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=1e-5, total_iters=config.train_config["epochs"])
    elif scheduler == "Exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.9)

    return scheduler

def initialise_configs(args):
    config = Config(args.config)
    if args.data_path:
        config.data_config["data_path"] = args.data_path
    if args.saved_model:
        config.data_config["saved_model"] = args.saved_model

    return config

def train_epoch(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()


    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
