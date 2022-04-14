import random
from collections import namedtuple

import torch
from torch._C import device

# from codebase.torchutils.common import auto_device, compute_flops

# following the setting in OFA
N_UNITS = 5
DEPTHS = [2, 3, 4]
N_LAYERS_PER_UNIT = max(DEPTHS)
N_DEPTHS = len(DEPTHS)
EXPAND_RATIOS = [3, 4, 6]
N_EXPAND_RATIOS = len(EXPAND_RATIOS)
KERNEL_SIZES = [3, 5, 7]
N_KERNEL_SIZES = len(KERNEL_SIZES)
AVAILABLE_RESOLUTIONS = [192, 208, 224, 240, 256]
N_AVAILABLE_RESOLUTIONS = len(AVAILABLE_RESOLUTIONS)

N_LAYERS = N_UNITS * N_LAYERS_PER_UNIT

combine2kernel_expand = {
    0: (0, 0),
    1: (3, 3),
    2: (3, 4),
    3: (3, 6),
    4: (5, 3),
    5: (5, 4),
    6: (5, 6),
    7: (7, 3),
    8: (7, 4),
    9: (7, 6),
}

kernel_expand2combine = dict((v, k) for k, v in combine2kernel_expand.items())


def split(items, separator=","):
    return [int(item) for item in items.split(separator)]


def join(items, separator=","):
    return separator.join(map(str, items))


def compute_depths(ks, ratios):
    for k, ratio in zip(ks, ratios):
        if k == 0:
            if ratio == 0:
                pass
            else:
                raise ValueError(f"The depth of ks ({join(ks)}) disagree with ratios ({join(ratios)}).")
    ks_ = torch.tensor(ks, dtype=torch.long)
    mask = torch.where(ks_ != 0, torch.tensor(1), torch.tensor(0))
    depths = mask.view(N_UNITS, N_LAYERS_PER_UNIT).long().sum(dim=1, keepdim=False).tolist()
    return depths


def adjust_ints(arch_ints):
    for n_unit in range(N_UNITS):
        third = arch_ints[n_unit*N_LAYERS_PER_UNIT+2]
        fourth = arch_ints[n_unit*N_LAYERS_PER_UNIT+3]
        if third == 0 and fourth != 0:
            arch_ints[n_unit*N_LAYERS_PER_UNIT+2] = fourth
            arch_ints[n_unit*N_LAYERS_PER_UNIT+3] = third
    return arch_ints


resolution_embeddins = torch.eye(n=N_AVAILABLE_RESOLUTIONS, dtype=torch.float, requires_grad=False)
ofa_one_hot_embeddings = torch.eye(n=len(combine2kernel_expand), dtype=torch.float, requires_grad=False)


class OFAArchitecture:
    def __init__(self, depths, ks, ratios, resolution=None):
        self.depths = depths
        self.ks = ks
        self.ratios = ratios
        self.resolution = resolution
        self.top1_acc = 0.0
        self.madds = 1000.0
        self.latency = 0.0
        self.prune()

        self._tensor = None

    def prune(self):
        cnt = 0
        for n_unit in range(N_UNITS):
            for n_layer in range(N_LAYERS_PER_UNIT):
                if n_layer >= self.depths[n_unit]:
                    self.ks[cnt] = 0
                    self.ratios[cnt] = 0
                cnt += 1

    @classmethod
    def random(cls, has_resolution=False):
        depths = random.choices(DEPTHS, k=N_UNITS)
        ks = random.choices(KERNEL_SIZES, k=N_UNITS*N_LAYERS_PER_UNIT)
        ratios = random.choices(EXPAND_RATIOS, k=N_UNITS*N_LAYERS_PER_UNIT)
        resolution = random.choice(AVAILABLE_RESOLUTIONS) if has_resolution else None
        return cls(depths, ks, ratios, resolution)

    @classmethod
    def from_legency_string(cls, arch_string):
        '''
        arch string example:
        3,4,4,3,3:5,7,5,0,7,7,7,3,3,5,3,7,7,5,5,0,7,7,5,0:4,3,4,0,6,6,6,6,4,6,6,6,3,4,3,0,6,3,3,0
        '''
        depths, ks, ratios = arch_string.split(":")
        return cls(split(depths), split(ks), split(ratios))

    def to_legency_string(self):
        '''
        arch string example:
        3,4,4,3,3:5,7,5,0,7,7,7,3,3,5,3,7,7,5,5,0,7,7,5,0:4,3,4,0,6,6,6,6,4,6,6,6,3,4,3,0,6,3,3,0
        '''
        return f"{join(self.depths)}:{join(self.ks)}:{join(self.ratios)}"

    @classmethod
    def from_string(cls, arch_string):
        '''
        arch string example
        5,7,5,0,9,9,9,3,2,6,3,9,7,5,4,0,9,7,4,0
        '''
        return cls.from_ints(split(arch_string))

    def to_string(self):
        '''
        arch string example
        5,7,5,0,9,9,9,3,2,6,3,9,7,5,4,0,9,7,4,0
        '''
        return join(self.to_ints())

    @classmethod
    def from_ints(cls, arch_ints):
        ks = []
        ratios = []

        resolution = AVAILABLE_RESOLUTIONS[arch_ints[0]] if len(arch_ints) > N_LAYERS else None
        new_arch_ints = adjust_ints(arch_ints[-N_LAYERS:])
        for combina_value in new_arch_ints:
            k, ratio = combine2kernel_expand[combina_value]
            ks.append(k)
            ratios.append(ratio)
        return cls(compute_depths(ks, ratios), ks, ratios, resolution)

    def to_ints(self):
        combines = []
        if self.resolution is not None:
            combines.append(AVAILABLE_RESOLUTIONS.index(self.resolution))
        for k, ratio in zip(self.ks, self.ratios):
            combines.append(kernel_expand2combine[(k, ratio)])
        return combines

    def to_tensor(self):
        if self._tensor is None:
            with torch.no_grad():
                embeddings = []
                if self.resolution is not None:
                    index = torch.tensor(self.to_ints()[0], dtype=torch.long)
                    embeddings.append(torch.index_select(resolution_embeddins, dim=0, index=index).flatten())
                index = torch.tensor(self.to_ints()[-N_LAYERS:], dtype=torch.long)
                embeddings.append(torch.index_select(ofa_one_hot_embeddings, dim=0, index=index).flatten())
                self._tensor = torch.cat(embeddings)
        return self._tensor

    # def obtain_acc_by(self, acc_pred):
    #     self.top1_acc = acc_pred(self.to_tensor().unsqueeze(0).to(device=auto_device)).view([]).item()

    # def obtain_madds_by(self, supernet, resolution=224):
    #     supernet.set_active_subnet(ks=self.ks, e=self.ratios, d=self.depths)
    #     ofa_childnet = supernet.get_active_subnet(preserve_weight=False)
    #     self.madds = compute_flops(ofa_childnet, (1, 3, resolution, resolution), list(ofa_childnet.parameters())[0].device)/1e6

    def apply(self, edit):
        arch_ints = self.to_ints()
        for index, target in edit:
            arch_ints[index] = target
        return self.from_ints(arch_ints)

    @classmethod
    def from_lstm(cls, arch_seq):
        return cls.from_ints(arch_seq)

    def __hash__(self):
        return hash(self.to_string())
