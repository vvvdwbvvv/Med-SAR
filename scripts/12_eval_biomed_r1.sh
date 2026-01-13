#!/bin/bash
set -euo pipefail

port=8001

modes=(
  "reasoning"
)

eval_dataset="zou-lab/BioMed-R1-Eval"
eval_benchmarks=(
  hle_biomed
  medbullets_op4
  medbullets_op5
  medmcqa
  medqa
  MedXpertQA
  mmlu_health_biology
  pubmedqa
  GPQA_Medical_test
  Lancet
  NEJM
)

TEMPERATURE=0.2
USE_CHAT_TEMPLATE=true
STRICT_PROMPT=true

for mode in "${modes[@]}"; do
  for benchmark in "${eval_benchmarks[@]}"; do
    cmd="PYTHONPATH=src python -m med_sar.eval.baselines \
      --eval_dataset $eval_dataset \
      --eval_benchmark $benchmark \
      --port $port \
      --batch_size 32 \
      --max_new_tokens 4096 \
      --temperature $TEMPERATURE"

    if [ "$STRICT_PROMPT" == "true" ]; then
      cmd="$cmd --strict_prompt"
    fi

    if [ "$mode" == "reasoning" ]; then
      cmd="$cmd --reasoning"
    fi

    if [ "$USE_CHAT_TEMPLATE" == "true" ]; then
      cmd="$cmd --use_chat_template"
    fi

    eval "$cmd"
  done
done


# PYTHONPATH=src python -m med_sar.eval.baselines --backend transformers --model models/doctor_sft --eval_dataset zou-lab/BioMed-R1-Eval --eval_benchmark medqa