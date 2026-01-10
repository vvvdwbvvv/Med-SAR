UV=uv
PYTHON=python3

setup:
	$(UV) venv
	$(UV) pip install -e ".[dev]"

sync:
	$(UV) pip sync uv.lock

preprocess:
	$(UV) run scripts/00_build_m23k_json.py \
	--output_dir data/processed/m23k \
	--seed 42 \
	--train_ratio 0.9 \
	--val_ratio 0.05 \
	--test_ratio 0.05
    && \
	$(UV) run scripts/01_build_mimic_corpus.py \
	--input data/raw/NOTEEVENTS.csv \
	--output_dir data/processed \
	--text_col TEXT \
	--chunk_chars 1200 \
	--chunk_overlap 200 \
	--min_note_chars 40 \
	--min_chunk_chars 200

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
	$(UV) run scripts/04_sft_doctor.py \
	--train data/processed/m23k_train.jsonl \
	--dev data/processed/m23k_val.jsonl \
	--base meta-llama/Llama-3.1-8B-Instruct \
	--out models/doctor_sft \
	--max_len 1024

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

