# merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_id = "meta-llama/Llama-3.2-3B-Instruct"   
adapter_id = "EddieTsai123/doctor_sft_v2"
out_dir = "runs/merged_doctor_sft_v2"

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base, adapter_id)
model = model.merge_and_unload()

tok.save_pretrained(out_dir)
model.save_pretrained(out_dir)
print("saved merged model to", out_dir)