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
	--test_ratio 0.05 \
    ;
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

V2_M23K_OUT=data/processed/m23k_v2
V2_MIMIC_OUT=data/processed/mimic_v2
V2_CALIB_OUT=configs/calibration.json
V2_OPERATORS_OUT=configs/operators.yaml
V2_PROXIES_OUT=configs/proxies.yaml
V2_GUARD_OUT=runs/fact_guard
V2_GUARD_SPEC=configs/fact_guard.yaml
V2_ADV_OUT=data/processed/adv/adv_train.jsonl
V2_ADV_GUARD_LOG=runs/adv_gen/guard_log.jsonl
V2_ADV_PROXIES=runs/adv_gen/post_guard_proxies.parquet
V2_LOOP_OUT=runs/selfplay
V2_BENCH_OUT=runs/benchmark

v2_m23k:
	$(UV) run scriptsv2/00_build_m23k_json.py \
	--output_dir $(V2_M23K_OUT) \
	--seed 42 \
	--train_ratio 0.1 \
	--val_ratio 0.05 \
	--test_ratio 0.05

v2_mimic:
	$(UV) run scriptsv2/01_build_mimic_corpus.py \
	--input data/raw/NOTEEVENTS.csv \
	--output_dir $(V2_MIMIC_OUT) \
	--text_col TEXT \
	--min_note_chars 40

v2_calibrate:
	$(UV) run scriptsv2/03_calibrate_operator_strength.py \
	--mimic_manifest $(V2_MIMIC_OUT)/mimic_manifest.parquet \
	--out $(V2_CALIB_OUT) \
	--operators_out $(V2_OPERATORS_OUT) \
	--proxies_out $(V2_PROXIES_OUT)

v2_guard:
	$(UV) run scriptsv2/04_fact_guard_build.py \
	--m23k $(V2_M23K_OUT)/m23k_wrapped_train.jsonl \
	--calibration $(V2_CALIB_OUT) \
	--out_dir $(V2_GUARD_OUT) \
	--config_out $(V2_GUARD_SPEC)

v2_smoke:
	$(UV) run scriptsv2/02_operator_smoke_test.py \
	--m23k $(V2_M23K_OUT)/m23k_wrapped_val.jsonl \
	--calibration $(V2_CALIB_OUT) \
	--guard_spec $(V2_GUARD_SPEC) \
	--out runs/smoke/operator_smoke.jsonl \
	--diff_out runs/smoke/samples_diff.txt

v2_sft:
	$(UV) run scriptsv2/04_sft_doctor.py \
	--train $(V2_M23K_OUT)/m23k_wrapped_train.jsonl \
	--dev $(V2_M23K_OUT)/m23k_wrapped_val.jsonl \
	--base meta-llama/Llama-3.2-3B-Instruct \
	--out runs/sft_doctor/model \
	--input_field x_wrapped

v2_adv_batch:
	$(UV) run scriptsv2/05_generate_adv_batch.py \
	--m23k $(V2_M23K_OUT)/m23k_wrapped_train.jsonl \
	--calibration $(V2_CALIB_OUT) \
	--guard_spec $(V2_GUARD_SPEC) \
	--out $(V2_ADV_OUT) \
	--guard_log_out $(V2_ADV_GUARD_LOG) \
	--proxies_out $(V2_ADV_PROXIES)

v2_selfplay:
	$(UV) run scriptsv2/06_loop_train.py \
	--train $(V2_M23K_OUT)/m23k_wrapped_train.jsonl \
	--dev $(V2_M23K_OUT)/m23k_wrapped_val.jsonl \
	--base_model meta-llama/Llama-3.2-3B-Instruct \
	--out_dir $(V2_LOOP_OUT) \
	--calibration $(V2_CALIB_OUT) \
	--guard_spec $(V2_GUARD_SPEC)

v2_benchmark:
	$(UV) run scriptsv2/09_controlled_shift_benchmark.py \
	--preds outputs/preds.parquet \
	--manifest $(V2_MIMIC_OUT)/mimic_manifest.parquet \
	--out_dir $(V2_BENCH_OUT)

v2_plots:
	$(UV) run scriptsv2/12_plot_frontier_and_appendix.py \
	--out_dir $(V2_BENCH_OUT) \
	--guard_stats $(V2_ADV_GUARD_LOG)

v2_selfplay_report:
	$(UV) run scriptsv2/10_selfplay_evolving_effect.py \
	--policy_logs $(V2_LOOP_OUT)/policy_selection_log.jsonl \
	--out_dir runs/selfplay_evolving \
	--baseline_dir runs/baseline_benchmark \
	--selfplay_dir $(V2_BENCH_OUT)

v2_pipeline: v2_m23k v2_mimic v2_calibrate v2_guard v2_smoke v2_sft v2_selfplay v2_benchmark v2_plots v2_selfplay_report
	@echo "v2 pipeline complete"

checkstyle:
	ruff check . --exclude test; ruff_check_status=$$?; \
	ruff format --check . --exclude test; ruff_format_status=$$?; \
	ruff check . --fix --exclude test; \
	ruff format . --exclude test; \
	if [ $$ruff_check_status -ne 0 ] || [ $$ruff_format_status -ne 0 ]; then \
	    exit 1; \
	fi
