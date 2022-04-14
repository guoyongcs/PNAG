import json
import torch
from torch.utils.data import Dataset
from core.mlp import arch2tensor
from core.controller import str2arch


class ArchitectureDataset(Dataset):
    def __init__(self, path, train, seed, proportion, width, platform, output_type=None):
        with open(path, "r") as f:
            self.archs = json.load(f)
        n_total = len(self.archs)
        g = torch.Generator()
        g.manual_seed(seed)
        index = torch.randperm(n_total, generator=g).tolist()
        n_samples = int(n_total * proportion)
        if train:
            index = index[:n_samples]
        else:
            index = index[n_samples:]
        split = [self.archs[i] for i in index]
        self.tensors = [arch2tensor(str2arch(arch["arch"])) for arch in split]
        self.accuracies = [arch["w"+str(width)]["top1_acc"] for arch in split]
        self.latencies = [arch["w"+str(width)][platform+"_latency"] for arch in split]

        self.output_type = output_type

    def __getitem__(self, index):
        if self.output_type is None:
            return self.tensors[index], self.accuracies[index], self.latencies[index]
        elif self.output_type == "accuracy":
            return self.tensors[index], self.accuracies[index]
        else:
            return self.tensors[index], self.latencies[index]

    def __len__(self):
        return len(self.tensors)
