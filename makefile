UV=uv
PYTHON=python

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

sync:
	$(UV) pip sync uv.lock

smoke:
	$(UV) run python scripts/03_prompt_rewrite_smoke_test.py

train_critic:
	$(UV) run med-sar --task train_critic --config configs/default.yaml

sft_doctor:
	$(UV) run med-sar --task sft_doctor --config configs/default.yaml

loop:
	$(UV) run med-sar --task loop --config configs/default.yaml
