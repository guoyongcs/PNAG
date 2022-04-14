import json
import random
from itertools import combinations

import torch
import torch.nn as nn

from utils.metrics import AverageMetric
from utils import output_directory, logger

GRAD_CLIP_NORM = 1


def correct_nan(optimizer):
    def _correct_nan(grad):
        grad[torch.isnan(grad)] = 0
        return grad
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is None:
                continue
            grad = _correct_nan(p.grad.data)
            p.grad.data.copy_(grad)


def compute_rankloss_labels(accuracies, latencies, target_latency):
    labels = []

    for (A1, A2), (L1, L2) in zip(combinations(accuracies, r=2), combinations(latencies, r=2)):
        if L1 > target_latency and L2 > target_latency:
            labels.append(1 if L1 < L2 else -1)
        elif L1 <= target_latency and L2 <= target_latency:
            labels.append(1 if A1 > A2 else -1)
        else:
            labels.append(1 if L1 < L2 else -1)
    return labels


def pairwise_rankloss(predictions, accuracies, latencies, target_latency):
    prediction_combination = torch.combinations(predictions)
    labels = compute_rankloss_labels(accuracies, latencies, target_latency)

    labels = torch.tensor(labels, dtype=predictions.dtype, device=predictions.device)
    prediction_combination_left, prediction_combination_right = prediction_combination.unbind(dim=1)
    loss = (prediction_combination_left-prediction_combination_right)*labels

    loss = torch.log(1+torch.exp(-loss))
    loss = loss.sum()

    cnt = labels.abs().sum().item()
    loss = loss/(cnt+1e-8)
    return loss, cnt == 0


def compute_ktau(predictions, accuracies, latencies, target_latency):
    labels = compute_rankloss_labels(accuracies, latencies, target_latency)
    sequence = []
    for (P1, P2), label in zip(combinations(predictions, r=2), labels):
        sequence.append(label * (1 if P1 > P2 else -1))
    sequence = torch.tensor(sequence, dtype=float)
    return (sequence.sum()/(sequence.abs().sum()+1e-8)).item()


def latency_onehot_embedding(target_latency, target_latencies, batch_size):
    embedding = torch.zeros((batch_size, len(target_latencies)))
    index = target_latencies.index(target_latency)
    embedding[:, index] = 1
    return embedding


def train_one_epoch(epoch, loader, predictor, optimizer, scheduler, target_latencies, device):
    predictor.train()
    loss_metric = AverageMetric()
    outputs_list = []
    for iter_, (inputs, accuraries, latencies) in enumerate(loader):
        latencies = latencies * 1000
        target_latency = random.choice(target_latencies)
        inputs = inputs.to(device=device)
        conditions = latency_onehot_embedding(target_latency, target_latencies, inputs.size(0))
        conditions = conditions.to(device=device)
        inputs = torch.cat([inputs, conditions], dim=1)
        accuraries = accuraries.float().to(device=device)
        outputs = predictor(inputs)
        outputs = outputs.squeeze()
        loss, flag = pairwise_rankloss(outputs, accuraries.tolist(),
                                       latencies.tolist(), target_latency)
        if loss.item() <= 1e-3:
            print(outputs)
        if not flag:
            loss_metric.update(loss)

        optimizer.zero_grad()
        loss.backward()
        correct_nan(optimizer)
        nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP_NORM)

        if not flag:
            optimizer.step()

        outputs_list.append(outputs.detach())
    scheduler.step()
    return loss_metric.compute(), torch.cat(outputs_list, dim=-1)


def validate_one_epoch(epoch, loader, predictor, target_latencies, device):
    predictor.eval()
    ktau_metric = AverageMetric()
    loss_metric = AverageMetric()
    outputs_list = []
    with torch.no_grad():
        for iter_, (inputs, accuraries, latencies) in enumerate(loader):
            latencies = latencies * 1000
            for target_latency in target_latencies:
                inputs = inputs.to(device=device)
                conditions = latency_onehot_embedding(target_latency, target_latencies, inputs.size(0))
                conditions = conditions.to(device=device)

                inputs_ = torch.cat([inputs, conditions], dim=1)
                accuraries = accuraries.float().to(device=device)
                outputs = predictor(inputs_)
                outputs = outputs.squeeze()

                loss, flag = pairwise_rankloss(outputs, accuraries.tolist(),
                                               latencies.tolist(), target_latency,)
                if not flag:
                    loss_metric.update(loss)
                ktau = compute_ktau(outputs.tolist(), accuraries.tolist(), latencies.tolist(), target_latency)
                ktau_metric.update(ktau)
                outputs_list.append(outputs.detach())

    return ktau_metric.compute(), torch.cat(outputs_list, dim=-1), loss_metric.compute()


def train(max_epochs, train_loader, val_loader, reward_estimator, optimizer, scheduler, target_latencies, device):
    for epoch in range(max_epochs):
        train_loss, _ = train_one_epoch(epoch, train_loader, reward_estimator,
                                        optimizer, scheduler, target_latencies, device)
        ktau, outputs, val_loss = validate_one_epoch(epoch, val_loader, reward_estimator, target_latencies, device)
        logger.info(f"Complete epoch={epoch}, train_loss={train_loss:.8f}, val_loss={val_loss:.8f}, ktau={ktau:.2f}, "
                    + f"output max={outputs.max().item():.2f}, output min={outputs.min().item():.2f}, "
                    + f"output mean={outputs.mean().item():.2f}, output std={outputs.std().item():.2f}. ")

