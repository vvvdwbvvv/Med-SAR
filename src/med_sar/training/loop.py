import yaml
from med_sar.utils.seed import set_seed


def run_loop(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg["seed"])
    # TODO loop:
    # - sample clean x from m23k
    # - generate x_adv via G (prompt-based first; RL later)
    # - cycle reward via teacher embeddings; reject/penalize if drift
    # - critic score: C(x_adv) should look "real"
    # - doctor loss: D(x_adv) -> y (from m23k)
    # - update G and D (and optionally C on schedule)
    raise NotImplementedError
