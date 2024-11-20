from omegaconf import OmegaConf


def create_config(config_cls):
    return OmegaConf.structured(config_cls())

def to_dict(config):
    return OmegaConf.to_container(config, resolve=True)


