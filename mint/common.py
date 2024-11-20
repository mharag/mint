from omegaconf import OmegaConf


def create_config(config_object):
    return OmegaConf.structured(config_object)

def load_yaml_config(path):
    return OmegaConf.load(path)

def to_dict(config):
    return OmegaConf.to_container(config, resolve=True)


