import importlib
import torch


def read_py(path):
    spec = importlib.util.spec_from_file_location(name="module.name", location=path)
    config_dict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_dict)
    return config_dict


def get_device(lightning=False):
    if lightning:
        return 'gpu' if torch.cuda.is_available() else 'cpu'
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
