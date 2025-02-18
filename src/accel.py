# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.bfloat16)
#model_ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", torch_dtype=torch.bfloat16).to("cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", 
                          logging_steps=10, bf16 = True, 
                          gradient_checkpointing=True, 
                          per_device_train_batch_size=1, 
                          dataset_num_proc=40)
trainer = DPOTrainer(model=model,
                     #ref_model=model_ref, 
                     args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()