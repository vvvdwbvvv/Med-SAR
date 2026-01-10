UV=uv
PYTHON=python3

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

checkstyle:
	$(PYTHON) -m ruff check . --exclude test; ruff_check_status=$$?; \
	$(PYTHON) -m ruff format --check . --exclude test; ruff_format_status=$$?; \
	$(PYTHON) -m ruff check . --fix --exclude test; \
	$(PYTHON) -m ruff format . --exclude test; \
	if [ $$ruff_check_status -ne 0 ] || [ $$ruff_format_status -ne 0 ]; then \
	    exit 1; \
	fi