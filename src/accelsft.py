# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
def preprocess_function2(examples):
    parts = examples['accept'].split('<think>')
    partsrejected = examples['reject'].split('<think>')
    if (len(parts) < 2 or len(parts) > 2 or len(partsrejected) < 2 or len(partsrejected) > 2):
        return None
    assistantparts = examples['accept'].split("<｜Assistant｜>")
    if (len(assistantparts) > 2):
        return None
    examples['prompt'] = parts[0]
    examples['chosen'] = parts[1]
    
    examples['rejected'] = "<think>" + partsrejected[1]
    return examples

modelid = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#modelid = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(modelid)
#model_ref = AutoModelForCausalLM.from_pretrained(modelid).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(modelid)
#train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

ds = load_dataset("Hankbeasley/polycoder")
ds = ds['train'].map(preprocess_function2, batched=False)
train_dataset = ds
ds = ds.filter(lambda x: (len(x['chosen'])<35000 and len(x['rejected'])<35000))
ds = ds.remove_columns(["accept", "reject", "testname"])
split_dataset = ds.train_test_split(test_size=0.2)
training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", 
                          logging_steps=10,  
                          gradient_checkpointing=True, 
                          per_device_train_batch_size=15, 
                          dataset_num_proc=15,
                          gradient_accumulation_steps=4,
                          bf16=True)
trainer = DPOTrainer(model=model,
                     #ref_model=model_ref, 
                     args=training_args, processing_class=tokenizer, train_dataset=ds)
trainer.train()