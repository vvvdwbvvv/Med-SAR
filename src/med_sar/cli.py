import argparse
from med_sar.training.train_critic import run_train_critic
from med_sar.training.sft_doctor import run_sft_doctor
from med_sar.training.loop import run_loop


def main():
    p = argparse.ArgumentParser("med-sar")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train_critic", "sft_doctor", "loop"],
    )
    args = p.parse_args()

    if args.task == "train_critic":
        run_train_critic(args.config)
    elif args.task == "sft_doctor":
        run_sft_doctor(args.config)
    elif args.task == "loop":
        run_loop(args.config)
