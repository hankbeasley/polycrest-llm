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
    
    examples['chosen'] = "<think>" + parts[1]
    
    examples['rejected'] = "<think>" + partsrejected[1]
    return examples

modelid = "Hankbeasley/PolycrestSFT-Qwen-7B"
#modelid = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(modelid, torch_dtype=torch.bfloat16)
# model_ref = AutoModelForCausalLM.from_pretrained(modelid, torch_dtype=torch.bfloat16,
#  #attn_implementation="flash_attention_2"
#  )
tokenizer = AutoTokenizer.from_pretrained(modelid)
#train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

ds = load_dataset("Hankbeasley/polycoder")
ds = ds['train'].map(preprocess_function2, batched=False)
train_dataset = ds
ds = ds.filter(lambda x: (len(x['chosen'])<35000 and len(x['rejected'])<35000))
ds = ds.remove_columns(["accept", "reject", "testname"])

#ds = ds.select(range(100))

split_dataset = ds.train_test_split(test_size=0.2)


training_args = DPOConfig (output_dir="Qwen2-0.5B-DPO", 
                          logging_steps=10,  
                          gradient_checkpointing=True, 
                          per_device_train_batch_size=1, 
                          dataset_num_proc=30,
                          precompute_ref_log_probs=True,
                          precompute_ref_batch_size=1,
                          gradient_accumulation_steps=1,
                          max_length=None,
                          max_prompt_length=None,
                          max_completion_length=None,
                          bf16=True)

trainer = DPOTrainer(model=model,
                     #ref_model=model_ref, 
                     args=training_args, processing_class=tokenizer, train_dataset=ds)
trainer.accelerator.prepare_model(model)
a = trainer.get_train_dataloader()
print(a)
print(trainer.train_dataset)
trainer.train_dataset.push_to_hub("Hankbeasley/testds")
#trainer.train()