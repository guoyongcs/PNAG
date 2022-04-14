import os
import torch
import torch.nn.functional as F
from utils.metrics import AverageMetric
from utils import output_directory, logger

def train_one_epoch(epoch, loader, model, optimizer, scheduler, device):
    model.train()
    loss_metric = AverageMetric()
    for iter_, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device=device)
        targets = targets.float().to(device=device)

        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = F.mse_loss(outputs, targets)
        loss_metric.update(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    return loss_metric.compute()


def validate_one_epoch(epoch, loader, model, device):
    model.eval()
    diff_metric = AverageMetric()
    L2_loss_metric = AverageMetric()
    with torch.no_grad():
        for iter_, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device=device)
            targets = targets.float().to(device=device)

            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = F.l1_loss(outputs, targets)
            diff_metric.update(loss)
            L2_loss = F.mse_loss(outputs, targets)
            L2_loss_metric.update(L2_loss)
    return diff_metric.compute(), L2_loss_metric.compute()


def train(max_epochs, train_loader, val_loader, model, optimizer, scheduler, device):
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(epoch, train_loader, model, optimizer, scheduler, device)
        diff, val_loss = validate_one_epoch(epoch, val_loader, model, device)
        logger.info(f"epoch={epoch}, train_loss={train_loss:.8f}, val_loss={val_loss:.8f}, "
                    + f"val_diff={diff:.8f}.")
