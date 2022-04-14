import random

import torch

from core.mlp import MLP, arch2tensor, accuracy_predictor, latency_predictor
from core.controller import Controller, arch2config, arch2str

from utils import output_directory, logger, summary_writer
from utils.metrics import AverageMetric

from .reward_estimator import latency_onehot_embedding

REWARD_SCALE = 1e-3

def reward_estimator_infer(arch_tensor, target_latency, target_latencies, reward_estimator):
    with torch.no_grad():
        conditions = latency_onehot_embedding(target_latency, target_latencies, arch_tensor.size(0))
        conditions = conditions.to(device=arch_tensor.device)
        inputs = torch.cat([arch_tensor, conditions], dim=1)
        reward = reward_estimator(inputs).item()
        return reward

def compute_policy_loss(reward, log_p, entropy, entropy_coeff):
    policy_loss = -log_p*(reward*REWARD_SCALE)-entropy_coeff*entropy
    return policy_loss

def reward_mnas(arch_tensor, accuracy, latency, target_latency, w=-0.07):
    return accuracy*((latency/target_latency)**w)

def reward_tunas(arch_tensor, accuracy, latency, target_latency, beta=-0.07):
    return accuracy+beta*abs((latency/target_latency)-1)

def train_one_iteration(controller, optimizer, target_latencies, reward_estimator, entropy_coeff, device, 
                        method, latency_p, accuracy_p):
    controller.train()

    index = random.randint(0, len(target_latencies)-1)
    condition_tensor = torch.tensor([index], dtype=torch.long, device=device)
    arch, log_p, entropy = controller(condition_tensor)
    arch_tensor = arch2tensor(arch, unsqueeze=True).to(device=device)
    if method == "pfnas":
        reward = reward_estimator_infer(arch_tensor, target_latencies[index], target_latencies, reward_estimator)
    elif method == "mnasnet":
        latency = latency_p(arch_tensor).view([]).item()*1e3
        accuracy = accuracy_p(arch_tensor).view([]).item()
        reward = reward_mnas(arch_tensor, accuracy, latency, target_latencies[index])
    elif method == "tunas":
        latency = latency_p(arch_tensor).view([]).item()*1e3
        accuracy = accuracy_p(arch_tensor).view([]).item()
        reward = reward_tunas(arch_tensor, accuracy, latency, target_latencies[index])
    else:
        raise ValueError("Invalid reward method!")

    policy_loss = compute_policy_loss(reward, log_p, entropy, entropy_coeff)

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return reward, target_latencies[index]


def validate(iteration, n_test, controller, target_latencies,
             latency_p, accuracy_p, history_collection, last_iteration, device):
    controller.eval()

    with torch.no_grad():
        for condition_index in range(len(target_latencies)):
            entropy_metric = AverageMetric()
            logp_metric = AverageMetric()
            latency_metric = AverageMetric()
            diff_metric = AverageMetric()
            accuracy_metric = AverageMetric()
            target = target_latencies[condition_index]
            best_arch = None
            best_acc = 0
            best_latency = 0
            count = 0
            sample_latencies = []
            for _ in range(n_test):
                condition_tensor = torch.tensor([condition_index], dtype=torch.long, device=device)
                arch, log_p, entropy = controller(condition_tensor)

                arch_tensor = arch2tensor(arch).unsqueeze(0).to(device=device)

                latency = latency_p(arch_tensor).view([]).item()*1e3
                accuracy = accuracy_p(arch_tensor).view([]).item()

                entropy_metric.update(entropy)
                logp_metric.update(log_p)
                diff_metric.update(latency-target)
                accuracy_metric.update(accuracy*100)
                latency_metric.update(latency)

                sample_latencies.append(latency)

                flag = latency < target
                if accuracy > best_acc and flag:
                    best_acc = accuracy
                    best_latency = latency
                    best_arch = arch2str(arch)

                count += 1 if flag else 0
            if best_arch is not None:
                history_acc = history_collection[condition_index]["acc"]
                if best_acc > history_acc:
                    history_collection[condition_index] = dict(latency=target, predict_latency=best_latency, arch=best_arch, acc=best_acc)
            summary_writer.add_scalar(f"L{target}/avg_entropy", entropy_metric.compute(), iteration)
            summary_writer.add_scalar(f"L{target}/avg_log_p", logp_metric.compute(), iteration)
            summary_writer.add_scalar(f"L{target}/avg_diff", diff_metric.compute(), iteration)
            summary_writer.add_scalar(f"L{target}/avg_accuracy", accuracy_metric.compute(), iteration)
            summary_writer.add_scalar(f"L{target}/avg_latency", latency_metric.compute(), iteration)
            summary_writer.add_scalar(f"L{target}/best_accuracy", best_acc, iteration)
            summary_writer.add_scalar(f"L{target}/best_latency", best_latency, iteration)
            summary_writer.add_scalar(f"L{target}/latency_satisfied_rate", count/n_test, iteration)
            logger.info(f"iteration={iteration}, target={target}, entropy={entropy_metric.compute():.2f}, " +
                        f"logp={logp_metric.compute():.2f}, diff={diff_metric.compute():.2f}, accuracy={accuracy*100:.2f}%, latency={latency_metric.compute():.2f}")


def train(max_iterations, controller, optimizer, target_latencies, reward_estimator,
          latency_p, accuracy_p, entropy_coeff, eval_freq, device, method):
    history_collection = [dict(latency=0, arch="", acc=0.0) for _ in target_latencies]
    for iteration in range(max_iterations):
        reward, target_latency = train_one_iteration(controller, optimizer, target_latencies, reward_estimator, 
        entropy_coeff, device, method, latency_p, accuracy_p)
        summary_writer.add_scalar(f"L{target_latency}/reward", reward, iteration)
        if iteration % eval_freq == 0:
            n_test = 10
            validate(iteration, n_test, controller, target_latencies,
                     latency_p, accuracy_p, history_collection,
                     iteration == max_iterations-1, device)
    return history_collection
