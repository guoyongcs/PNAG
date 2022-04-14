import torch
import torch.nn as nn
from core.controller import Arch, N_UNITS, DEPTHS


def _zero(layers, items, n_units=N_UNITS, n_layers=max(DEPTHS)):
    cnt = 0
    for n_unit in range(n_units):
        for n_layer in range(n_layers):
            items[cnt] = 0 if n_layer >= layers[n_unit] else items[cnt]
            cnt += 1
    return items


def arch2tensor(arch: Arch, unsqueeze=False):
    ks = _zero(arch.depths, arch.ks)
    ratios = _zero(arch.depths, arch.ratios)
    tensor = []
    for k in ks:
        if k == 0:
            tensor += [0, 0, 0]
        elif k == 3:
            tensor += [1, 0, 0]
        elif k == 5:
            tensor += [0, 1, 0]
        elif k == 7:
            tensor += [0, 0, 1]
        else:
            raise ValueError()
    for r in ratios:
        if r == 0:
            tensor += [0, 0, 0]
        elif r == 3:
            tensor += [1, 0, 0]
        elif r == 4:
            tensor += [0, 1, 0]
        elif r == 6:
            tensor += [0, 0, 1]
        else:
            raise ValueError()
    out = torch.tensor(tensor, dtype=torch.float)
    if unsqueeze:
        out = out.unsqueeze(dim=0)
    return out


def linear_bn_relu(in_features, out_features, bn=True, relu=True):
    layers = [nn.Linear(in_features, out_features)]
    if bn:
        layers.append(nn.BatchNorm1d(out_features))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return layers


class MLP(nn.Sequential):
    def __init__(self, in_features, hidden_features, out_features, n_layers,
                 batchnorm=False, force_last_batchnorm=False, affine=False):
        assert n_layers > 1
        layers = []
        for i in range(n_layers):
            in_f = in_features if i == 0 else hidden_features
            out_f = out_features if i == n_layers-1 else hidden_features
            bn = batchnorm and i != n_layers-1
            relu = i != n_layers-1
            layers += linear_bn_relu(in_f, out_f, bn, relu)
        if force_last_batchnorm:
            layers += [nn.BatchNorm1d(out_features, affine=affine)]
        super(MLP, self).__init__(*layers)


def accuracy_predictor(width, pretrained=True,):
    # following the setting in OFA
    p = MLP(20*3*2, 512, 1, 3)
    if pretrained:
        p.load_state_dict(torch.load(f"asserts/w{width}_top1_acc_mlp.pth", map_location="cpu"))
    return p


def latency_predictor(width, latency_device, pretrained=True):
    # following the setting in OFA
    p = MLP(20*3*2, 512, 1, 3)
    if pretrained:
        p.load_state_dict(torch.load(f"asserts/w{width}_{latency_device}_latency_mlp.pth",
                                     map_location="cpu"))
    return p
