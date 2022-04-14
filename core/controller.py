import torch.nn as nn
import functools
import torch.nn.functional as F
import torch
from collections import namedtuple

# following the setting in OFA
N_UNITS = 5
DEPTHS = [2, 3, 4]
N_DEPTHS = len(DEPTHS)
EXPAND_RATIOS = [3, 4, 6]
N_EXPAND_RATIOS = len(EXPAND_RATIOS)
KERNEL_SIZES = [3, 5, 7]
N_KERNEL_SIZES = len(KERNEL_SIZES)

Arch = namedtuple("Arch", ["depths", "ks", "ratios"])


class Controller(nn.Module):
    def __init__(self, n_conditions=1, n_unit=N_UNITS,
                 depths=DEPTHS, kernel_sizes=KERNEL_SIZES, expand_ratios=EXPAND_RATIOS,
                 hidden_size=64, batch_size=1, device="cpu"):
        super(Controller, self).__init__()
        self.n_unit = n_unit
        self.depths = depths
        self.expand_ratios = expand_ratios
        self.kernel_sizes = kernel_sizes

        self.hidden_size = hidden_size

        self.condition_embedding = nn.Embedding(n_conditions, self.hidden_size)

        self.depth_embedding = nn.Embedding(len(self.depths), self.hidden_size)
        self.ratio_embedding = nn.Embedding(len(self.expand_ratios), self.hidden_size)
        self.ks_embedding = nn.Embedding(len(self.kernel_sizes), self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.depth_linear = nn.Linear(self.hidden_size, len(self.depths))
        self.width_linear = nn.Linear(self.hidden_size, len(self.expand_ratios))
        self.ks_linear = nn.Linear(self.hidden_size, len(self.kernel_sizes))

        self.batch_size = batch_size
        self.device = device
        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.device, requires_grad=False)

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def forward(self, condition=None, uniform=False):
        log_ps = []
        entrpys = []
        if condition is None:
            inputs = self._zeros(self.batch_size)
        else:
            inputs = self.condition_embedding(condition)

        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = inputs

        depths = []
        ks = []
        ratios = []

        for unit in range(self.n_unit):
            # depth
            if uniform:
                logits = torch.zeros(len(self.depths))
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.depth_linear(hx)
            probs = F.softmax(logits, dim=-1)
            depth, log_p, entropy = self._impl(probs)
            log_ps.append(log_p)
            entrpys.append(entropy)

            depths.append(self.depths[depth.item()])

            embed = self.depth_embedding(depth)

            for _ in range(max(self.depths)):
                # expand ratio
                if uniform:
                    logits = torch.zeros(len(self.expand_ratios))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.width_linear(hx)
                probs = F.softmax(logits, dim=-1)
                ratio, log_p, entropy = self._impl(probs)
                log_ps.append(log_p)
                entrpys.append(entropy)

                ratios.append(self.expand_ratios[ratio.item()])

                embed = self.ratio_embedding(ratio)

                # kernel_size
                if uniform:
                    logits = torch.zeros(len(self.kernel_sizes))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.ks_linear(hx)
                probs = F.softmax(logits, dim=-1)
                k, log_p, entropy = self._impl(probs)
                log_ps.append(log_p)
                entrpys.append(entropy)

                ks.append(self.kernel_sizes[k.item()])

                embed = self.ks_embedding(k)

        return Arch(depths, ks, ratios), sum(log_ps), sum(entrpys)


def str2arch(string):
    def split(items, separator=","):
        return [int(item) for item in items.split(separator)]
    depths_str, ks_str, ratios_str = string.split(":")
    return Arch(split(depths_str), split(ks_str), split(ratios_str))


def arch2str(arch: Arch):
    def join(items, separator=","):
        return separator.join(map(str, items))
    return f"{join(arch.depths)}:{join(arch.ks)}:{join(arch.ratios)}"


def arch2config(arch: Arch):
    from collections import OrderedDict
    config = OrderedDict()
    config["name"] = "MobileNetV3"
    config["bn"] = {
        "momentum": 0.1,
        "eps": 1e-05
    }
    config["first_conv"] = {
        "name": "ConvLayer",
        "kernel_size": 3,
        "stride": 2,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "has_shuffle": False,
        "in_channels": 3,
        "out_channels": 16,
        "use_bn": True,
        "act_func": "h_swish",
        "dropout_rate": 0,
        "ops_order": "weight_bn_act",
    }
    blocks = [
        {
            "name": "MobileInvertedResidualBlock",
            "mobile_inverted_conv": {
                "name": "MBInvertedConvLayer",
                "in_channels": 16,
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 1,
                "expand_ratio": 1,
                "mid_channels": None,
                "act_func": "relu",
                "use_se": False
            },
            "shortcut": {
                "name": "IdentityLayer",
                "in_channels": 16,
                "out_channels": 16,
                "use_bn": False,
                "act_func": None,
                "dropout_rate": 0,
                "ops_order": "weight_bn_act"
            }
        },
    ]
    stride_stages = [2, 2, 2, 1, 2]
    act_stages = ['relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
    se_stages = [False, True, False, True, True]
    widths = [16, 24, 40, 80, 112, 160]
    for unit_idx, n_layers in enumerate(arch.depths):
        # n_layers = unit["n_layers"]
        for layer_idx in range(n_layers):
            s = stride_stages[unit_idx] if layer_idx == 0 else 1
            in_c = widths[unit_idx] if layer_idx == 0 else widths[unit_idx+1]
            out_c = widths[unit_idx+1]
            skip = s == 1 and in_c == out_c
            block = OrderedDict()
            block["name"] = "MobileInvertedResidualBlock"
            block["mobile_inverted_conv"] = {
                "name": "MBInvertedConvLayer",
                "in_channels": in_c,
                "out_channels": out_c,
                "kernel_size": arch.ks[unit_idx*4+layer_idx],
                "stride": s,
                "expand_ratio": arch.ratios[unit_idx*4+layer_idx],
                "mid_channels": None,
                "act_func": act_stages[unit_idx],
                "use_se": se_stages[unit_idx]
            }
            if skip:
                block["shortcut"] = {
                    "name": "IdentityLayer",
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "use_bn": False,
                    "act_func": None,
                    "dropout_rate": 0,
                    "ops_order": "weight_bn_act"
                }
            else:
                block["shortcut"] = None

            blocks.append(block)

    config["blocks"] = blocks
    config["final_expand_layer"] = {
        "name": "ConvLayer",
        "kernel_size": 1,
        "stride": 1,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "has_shuffle": False,
        "in_channels": 160,
        "out_channels": 960,
        "use_bn": True,
        "act_func": "h_swish",
        "dropout_rate": 0,
        "ops_order": "weight_bn_act"
    }
    config["feature_mix_layer"] = {
        "name": "ConvLayer",
        "kernel_size": 1,
        "stride": 1,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "has_shuffle": False,
        "in_channels": 960,
        "out_channels": 1280,
        "use_bn": False,
        "act_func": "h_swish",
        "dropout_rate": 0,
        "ops_order": "weight_bn_act"
    }
    config["classifier"] = {
        "name": "LinearLayer",
        "in_features": 1280,
        "out_features": 1000,
        "bias": True,
        "use_bn": False,
        "act_func": None,
        "dropout_rate": 0.1,
        "ops_order": "weight_bn_act"
    }
    return config
