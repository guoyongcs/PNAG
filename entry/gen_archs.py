import sys
import json
import torch
import torch.nn
from dataclasses import dataclass
from utils.typed_args import TypedArgs, add_argument
from core.controller import Controller, arch2str
import typing

@dataclass
class Args(TypedArgs):
    n_samples: int = add_argument('--n_samples', default=3)
    out_file: str = add_argument("--out_file", default=None)
    training_condition_latencies: str = add_argument('--training_condition_latencies', default=None)
    target_latencies: str = add_argument('--target_latencies', default=None)
    controller_state_dict: str = add_argument('--controller_state_dict', default=None)
    device: str = add_argument('--device', default=None)


args = Args.from_known_args(sys.argv)


def interpolation(known_latencies, new_latency:int, embedding:torch.nn.Embedding):

    # 已知的延迟时间点和对应的索引
    known_latencies = torch.tensor(known_latencies)
    indices = torch.tensor(list(range(len(known_latencies))))


    # 计算插值位置
    pos = torch.searchsorted(known_latencies, new_latency) - 1
    pos = pos.clamp(0, len(known_latencies) - 2)  # 防止越界

    # 获取相邻的索引
    left_idx = indices[pos]
    right_idx = indices[pos + 1]

    # 计算权重
    left_weight = (known_latencies[pos + 1] - new_latency) / (known_latencies[pos + 1] - known_latencies[pos])
    right_weight = (new_latency - known_latencies[pos]) / (known_latencies[pos + 1] - known_latencies[pos])
    
    # 获取相应的嵌入
    left_emb = embedding(left_idx)
    right_emb = embedding(right_idx)

    # 计算插值后的嵌入
    interpolated_emb:torch.Tensor = left_weight * left_emb + right_weight * right_emb

    return interpolated_emb.unsqueeze(0)
    # print("Interpolated Embedding:", interpolated_emb)

if __name__ == "__main__":
    assert args.out_file != None
    training_condition_latencies = list(map(int, args.training_condition_latencies.split(",")))
    target_latencies = list(map(int, args.target_latencies.split(",")))
    device = args.device

    controller = Controller(n_conditions=len(training_condition_latencies), device=device).to(device=device)

    if args.controller_state_dict is not None:
        state_dict = torch.load(args.controller_state_dict, map_location=device)
        controller.load_state_dict(state_dict)

    results = {}
    
    for latency in target_latencies: 
        archs = set()
        while len(archs) < args.n_samples:
            with torch.no_grad():
                interpolated_emb= interpolation(
                    known_latencies=training_condition_latencies,
                    new_latency=latency,
                    embedding=controller.condition_embedding
                )
                arch = arch2str(controller(embed=interpolated_emb)[0])
            archs.add(arch)
            print(f"generating for latency {latency}, No. {len(archs)}/{args.n_samples}", arch)
        archs = list(archs)
        results[latency] = archs

    from pprint import pprint
    print("results:")
    pprint(results)
    with open(args.out_file, "w") as f:
        json.dump(archs, f)


'''
Example for running command:
python -m entry.gen_archs --target_latencies 45,55 --n_samples 1 --training_condition_latencies 30,40,50,60,70  --device cpu --out_file generated_archs.json --controller_state_dict xxx

参数解释：
target_latencies：目标的延迟
n_samples：每个目标延迟生成多少个网络结构
training_condition_latencies：训练controller所用的延迟列表
device：跑在cpu还是cuda上
out_file：结果输出到哪里，结果会使用json格式存储
controller_state_dict：训练好的controller的权重文件
'''