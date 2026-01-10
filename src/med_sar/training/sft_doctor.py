import yaml
from med_sar.utils.seed import set_seed


def run_sft_doctor(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    set_seed(cfg["seed"])
    # TODO:
    # 1) load m23k (question/reasoning/answer)
    # 2) SFT Doctor to produce reasoning+answer given question
    # 3) save to outputs/checkpoints/doctor_sft
    raise NotImplementedError
