import yaml
from med_sar.utils.seed import set_seed

def run_train_critic(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg["seed"])
    # TODO:
    # 1) load mimic real notes (label=1) + generated notes (label=0) from a cache
    # 2) train DistilBERT classifier
    # 3) save to outputs/checkpoints/critic
    raise NotImplementedError
