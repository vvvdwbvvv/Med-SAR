UV=uv
PYTHON=python3

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

sync:
	$(UV) pip sync uv.lock

smoke:
	$(UV) run scripts/02_prompt_rewrite_smoke_test.py \
	--m23k data/processed/m23k_val.jsonl \
	--out outputs/smoke/rewrite_samples.jsonl \
	--n 50 \
	--level 0.5 \
	--seed 42

train_critic:
	$(UV) run scripts/03_train_critic.py \
	--mimic_jsonl data/processed/corpus.jsonl \
	--m23k_dev data/processed/m23k_val.jsonl \
	--base distilbert/distilbert-base-uncased \
	--out models/critic \
	--n_pos 20000 \
	--n_neg 20000 \
	--level 0.3 \
	--seed 42

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

