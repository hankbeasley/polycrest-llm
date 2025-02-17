from threading import Thread
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from trl import (
    ModelConfig,
    DPOConfig,
    DPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from transformers import DataCollatorWithPadding
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase


def find_max_length(dataset, column_name):
    return max(len(tokens) for tokens in dataset[column_name])
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


if __name__ == "__main__":
    print("Starting...")
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # Load model on CPU
    
    from datasets import load_dataset

    ds = load_dataset("Hankbeasley/polycoder")
    ds = ds['train'].map(preprocess_function2, batched=False)
    print (ds)
    ds = ds.filter(lambda x: (len(x['chosen'])<35000 and len(x['rejected'])<35000))
    print (ds)
    ds = ds.remove_columns(["accept", "reject", "testname"])
    # Create train/test split (80% train, 20% test by default)
    split_dataset = ds.train_test_split(test_size=0.2)
    print(split_dataset)
    trainargs = DPOConfig (
        
        output_dir="/work/output",
        logging_dir="/work/output/logs",           # Directory to save logs
        logging_steps=50,                    # Log every 50 steps
        per_device_train_batch_size=15,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",         # Evaluate every few steps
        eval_steps=100,                      # Evaluate every 100 steps
        save_steps=100,                      # Save checkpoint every 500 steps
        report_to="tensorboard",
        #dataset_num_proc = 12,
        #push_to_hub=True,
        #hub_model_id="Hankbeasley/Polycrest-Qwen-1.5B",
        #push_to_hub_organization="hankbeasley",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable()
    print(model)
    from transformers import DataCollatorWithPadding

    #data_collator = DualDataCollator(tokenizer)
    
    trainer = DPOTrainer(

        model=model,
        processing_class=tokenizer,
        args=trainargs,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test']
        #data_collator=data_collator, 
        
        
    )
    trainer.train()
