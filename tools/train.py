import os
import sys
import json
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from core.controller import str2arch
from core.mlp import MLP, arch2tensor
from core.controller import Controller, arch2config, arch2str
from utils.metrics import AverageMetric
from utils import output_directory, logger
from utils.typed_args import TypedArgs, add_argument
from utils.common import set_reproducible

from core.dataset import ArchitectureDataset
from core.engine.predictor import train as train_predictor
from core.engine.reward_estimator import train as train_reward_estimator
from core.engine.controller import train as train_controller
from core.engine.evaluate_archs import evaluate_archs
device = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class Args(TypedArgs):
    seed: int = add_argument('--seed', default=1000)
    method: str = add_argument('--method', default="pfnas")
    data: str = add_argument('--data', default="assets/architectures.json",
                             help="The json file that stores the accuracy and latency of architectures")
    imagenet_path: str = add_argument('--imagenet_path', default=None,
                                      help="The path to ImageNet")

    width: str = add_argument('--width', default=1.2,
                              help="The initial width of the OFA supernet, [1.0, 1.2]")
    platform: str = add_argument('--platform', default=None,
                                 help="The platform for searching architectures, [cpu, gpu, mobile]")
    # parameter for accuracy/latency predictor
    p_epochs: int = add_argument('--p_epochs', default=1000,
                                 help="Trianing epochs for the accuracy/latency predictor")
    p_lr: float = add_argument('--p_lr', default=0.2,
                               help="Leraning rate for the accuracy/latency predictor")
    p_n_layers: int = add_argument('--p_n_layers', default=3,
                                   help="The number of layers in the accuracy/latency predictor")
    p_n_hidden: int = add_argument('--p_n_hidden', default=512,
                                   help="THe number of hidden nerons in the accuracy/latency predictor")

    condition_latencies: str = add_argument('--condition_latencies', default=None)

    r_epochs: int = add_argument('--r_epochs', default=200,
                                 help="The number of trianing epochs for the reward estimator.")

    c_lr: float = add_argument('--c_lr', default=3e-4,
                               help="Leraning rate for the controller.")
    c_iterations: int = add_argument('--c_iterations', default=10001,
                                     help="The number of trianing iterations for the controller.")
    entropy_coeff: float = add_argument('--entropy_coeff', default=1e-3,
                                        help="The entropy coeff for the controller.")
    eval_freq: int = add_argument('--eval_freq', default=500,
                                  help="Evaluation frequency while training the controller.")


args = Args.from_known_args(sys.argv)

if args.imagenet_path is not None:
    assert os.path.exists(os.path.join(args.imagenet_path, "train"))
    assert os.path.exists(os.path.join(args.imagenet_path, "val"))


def construct_model(in_features, n_hidden, n_layers, lr, max_epochs,
                    device, weight_decay=3e-5, momentum=0.9):
    model = MLP(in_features, n_hidden, 1, n_layers).to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          weight_decay=weight_decay, momentum=momentum, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=lr/1e4)
    return model, optimizer, scheduler


def construct_data(data, training_proportion, width, platform, output_type=None, batch_size=256, seed=2020, num_workers=0):
    trainset = ArchitectureDataset(data, True, seed, training_proportion, width, platform, output_type)
    valset = ArchitectureDataset(data, False, seed, training_proportion, width, platform, output_type)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    logger.info(args)
    set_reproducible(args.seed)
    # train the accuracy predictor
    logger.info("Start to train the accuracy predictor.")
    # following OFA, the network has 5 units, 4 layers at most per unit
    # each layers has two dimension (kernel size and channel expand ratio)
    # kernel size belongs to {3,5,7}, channel expand ratio belongs to {3,4,6}
    in_features = (4*5)*3*2
    accuracy_predictor, optimizer, scheduler = construct_model(in_features, args.p_n_hidden, args.p_n_layers,
                                                               args.p_lr, args.p_epochs, device)
    train_loader, val_loader = construct_data(args.data, 0.8, args.width, args.platform, "accuracy")

    train_predictor(args.p_epochs, train_loader, val_loader, accuracy_predictor, optimizer, scheduler, device)

    torch.save(accuracy_predictor.state_dict(), os.path.join(output_directory, "accuracy_predictor.path"))

    # train the latency predictor
    logger.info("Start to train the latency predictor.")
    latency_predictor, optimizer, scheduler = construct_model(in_features, args.p_n_hidden, args.p_n_layers,
                                                              args.p_lr, args.p_epochs, device)
    train_loader, val_loader = construct_data(args.data, 0.8, args.width, args.platform, "latency")

    train_predictor(args.p_epochs, train_loader, val_loader, latency_predictor, optimizer, scheduler, device)

    torch.save(latency_predictor.state_dict(), os.path.join(output_directory, "latency_predictor.path"))

    target_latencies = list(map(int, args.condition_latencies.split(",")))
    # train the reward estimator
    if args.method == "pfnas":
        logger.info("Start to train the reward estimator.")

        
        reward_estimator, optimizer, scheduler = construct_model(in_features+len(target_latencies), args.p_n_hidden, args.p_n_layers,
                                                                args.p_lr, args.p_epochs, device)
        train_loader, val_loader = construct_data(args.data, 0.8, args.width, args.platform)

        train_reward_estimator(args.r_epochs, train_loader, val_loader, reward_estimator,
                            optimizer, scheduler, target_latencies, device)

        torch.save(reward_estimator.state_dict(), os.path.join(output_directory, "reward_estimator.path"))
    elif args.method == "mnasnet" or args.method == "tunas":
        logger.info("Skip to train the reward estimator")
        reward_estimator = None
    else:
        raise ValueError("Invalid reward method!")

    # train the controller
    logger.info("Start to train the controller.")
    controller = Controller(n_conditions=len(target_latencies), device=device).to(device=device)
    optimizer = optim.Adam(controller.parameters(), lr=args.c_lr)
    searched_architectures = train_controller(args.c_iterations, controller, optimizer, target_latencies,
                                              reward_estimator, latency_predictor, accuracy_predictor,
                                              args.entropy_coeff, args.eval_freq, device, args.method)
    if args.imagenet_path is not None:
        evaluate_archs(searched_architectures, args.width, args.imagenet_path)
    logger.info("Searching architectures completes.")
    for item in searched_architectures:
        logger.info(item)
    
    torch.save(controller.state_dict(), os.path.join(output_directory, "controller.path"))
