from core.model.ofa_mbv3 import OFAMobileNetV3
from core.controller import str2arch, arch2str, Arch

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

WIDTH = 1.2
archs_strings = {
    'pnag-cpu-30': '2,2,3,3,3:5,5,0,0,5,5,0,0,7,5,7,0,3,5,5,0,7,5,5,0:3,3,0,0,4,3,0,0,4,3,3,0,6,4,4,0,6,6,6,0',
    'pnag-cpu-35': '2,3,3,3,3:5,3,0,0,5,5,7,0,5,7,7,0,5,5,7,0,7,5,7,0:3,4,0,0,4,6,3,0,6,4,6,0,6,6,6,0,6,6,6,0',
    'pnag-cpu-40': '2,3,4,4,4:5,3,0,0,7,5,7,0,5,5,3,3,5,7,7,7,3,3,3,7:4,4,0,0,4,4,4,0,4,6,6,4,6,4,6,6,6,6,6,4',
    'pnag-cpu-45': '2,3,4,4,4:7,5,0,0,7,7,7,0,7,5,3,7,7,7,7,7,7,3,3,7:4,6,0,0,6,6,6,0,6,6,6,6,6,6,6,6,6,6,6,6',
    'pnag-cpu-50': '3,4,4,4,4:5,3,7,0,5,7,7,7,7,7,3,7,7,7,7,7,5,7,3,7:4,4,6,0,6,6,6,6,6,6,4,6,6,6,6,6,6,6,6,6',
    #---------------------------------------------------------------------------------------------------------
    'pnag-gpu-90': '2,2,3,3,4:3,3,0,0,5,3,0,0,5,5,3,0,3,7,3,0,5,3,3,7:3,3,0,0,3,4,0,0,3,3,3,0,4,3,4,0,6,6,6,3',
    'pnag-gpu-115': '2,3,3,4,4:3,3,0,0,5,3,3,0,5,5,3,0,3,5,5,5,7,7,7,5:3,3,0,0,4,4,4,0,4,3,4,0,6,4,6,4,6,6,6,4',
    'pnag-gpu-140': '2,3,4,4,4:3,3,0,0,7,5,5,0,5,7,5,5,3,5,5,5,7,7,5,5:4,3,0,0,6,4,4,0,4,4,6,6,6,6,6,6,6,6,6,4',
    'pnag-gpu-165': '2,4,4,4,4:3,3,0,0,7,5,5,5,7,5,7,5,7,7,7,5,7,7,7,5:3,4,0,0,6,6,6,6,6,6,4,6,6,6,6,6,6,6,6,6',
    'pnag-gpu-190': '3,4,4,4,4:3,3,5,0,7,3,5,7,7,7,5,3,7,5,7,5,7,3,7,3:4,4,6,0,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6',
    #---------------------------------------------------------------------------------------------------------
    'pnag-mobile-80': '2,3,3,3,3:3,3,0,0,3,5,3,0,3,3,5,0,3,3,5,0,5,5,7,0:3,3,0,0,4,3,3,0,4,4,4,0,4,6,6,0,6,6,3,0',
    'pnag-mobile-110': '2,3,3,4,4:5,3,0,0,5,3,3,0,3,5,5,0,5,5,5,3,7,5,7,5:3,4,0,0,3,4,4,0,3,3,6,0,4,6,6,6,6,6,6,6',
    'pnag-mobile-140': '2,4,4,4,4:5,3,0,0,5,5,5,3,5,5,5,5,3,5,5,5,7,5,5,5:4,3,0,0,4,4,4,6,6,3,6,3,6,6,4,6,6,6,6,3',
    'pnag-mobile-170': '3,4,4,4,4:5,3,7,0,5,5,5,5,5,5,5,5,3,5,5,3,7,5,5,5:4,4,6,0,4,4,6,6,6,4,6,6,6,6,4,6,6,6,6,6',
    'pnag-mobile-200': '4,4,4,4,4:5,5,5,5,5,7,3,5,7,5,5,3,3,5,5,3,5,3,3,5:6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6',
}


archs_weights_urls = {
    "pnag-cpu-30": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-cpu-30-048f7ba8.pt",
    "pnag-cpu-35": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-cpu-35-a4f4b023.pt",
    "pnag-cpu-40": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-cpu-40-c3d1c578.pt",
    "pnag-cpu-45": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-cpu-45-899d30e2.pt",
    "pnag-cpu-50": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-cpu-50-394eb196.pt",
    #---------------------------------------------------------------------------------------------------------
    "pnag-gpu-90": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-gpu-90-990c2f4b.pt",
    "pnag-gpu-115": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-gpu-115-7df721b0.pt",
    "pnag-gpu-140": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-gpu-140-a0855635.pt",
    "pnag-gpu-165": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-gpu-165-74aa92f7.pt",
    "pnag-gpu-190": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-gpu-190-34e54866.pt",
    #---------------------------------------------------------------------------------------------------------
    "pnag-mobile-80": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-mobile-80-3fff8d1d.pt",
    "pnag-mobile-110": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-mobile-110-035a56f7.pt",
    "pnag-mobile-140": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-mobile-140-a0855635.pt",
    "pnag-mobile-170": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-mobile-170-ed35c80d.pt",
    "pnag-mobile-200": "https://github.com/guoyongcs/PNAG/releases/download/weights/pnag-mobile-200-ec1b53bf.pt",
}


def _get_pnag_from_supernet(name):
    net = OFAMobileNetV3(
        dropout_rate=0,
        width_mult_list=WIDTH,
        ks_list=[3, 5, 7],
        expand_ratio_list=[3, 4, 6],
        depth_list=[2, 3, 4],
    )
    arch: Arch = str2arch(archs_strings[name])
    net.set_active_subnet(ks=arch.ks, e=arch.ratios, d=arch.depths)
    subnet = net.get_active_subnet(preserve_weight=False)
    state_dict = load_state_dict_from_url(archs_weights_urls[name], progress=True)
    subnet.load_state_dict(state_dict)
    return subnet

def pnag_cpu_30():
        return _get_pnag_from_supernet("pnag-cpu-30")

def pnag_cpu_35():
        return _get_pnag_from_supernet("pnag-cpu-35")

def pnag_cpu_40():
        return _get_pnag_from_supernet("pnag-cpu-40")

def pnag_cpu_45():
        return _get_pnag_from_supernet("pnag-cpu-45")

def pnag_cpu_50():
        return _get_pnag_from_supernet("pnag-cpu-50")

def pnag_gpu_90():
        return _get_pnag_from_supernet("pnag-gpu-90")

def pnag_gpu_115():
        return _get_pnag_from_supernet("pnag-gpu-115")

def pnag_gpu_140():
        return _get_pnag_from_supernet("pnag-gpu-140")

def pnag_gpu_165():
        return _get_pnag_from_supernet("pnag-gpu-165")

def pnag_gpu_190():
        return _get_pnag_from_supernet("pnag-gpu-190")

def pnag_mobile_80():
        return _get_pnag_from_supernet("pnag-mobile-80")

def pnag_mobile_110():
        return _get_pnag_from_supernet("pnag-mobile-110")

def pnag_mobile_140():
        return _get_pnag_from_supernet("pnag-mobile-140")

def pnag_mobile_170():
        return _get_pnag_from_supernet("pnag-mobile-170")

def pnag_mobile_200():
        return _get_pnag_from_supernet("pnag-mobile-200")