import shutil
from threading import Thread
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TrainerCallback

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from transformers import DataCollatorWithPadding
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

class CustomStepCallback(TrainerCallback):
    def __init__(self, step_frequency: int, directory:str):
        self.step_frequency = step_frequency
        self.directory = directory

    def on_step_end(self, args, state, control, **kwargs):
        # Check if the current step is a multiple of the given frequency
        if state.global_step % self.step_frequency == 0:
           
            for item in os.listdir(self.directory):
                if (item.startswith("checkpoint")):
                    item_path = os.path.join(self.directory, item)
                    if os.path.isdir(item_path):
                        print("deleteing:" + item_path)
                        shutil.rmtree(item_path)
        return control



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

def preprocess_functionSFT(examples):
    parts = examples['accept'].split('<think>')
    if (len(parts) < 2 or len(parts) > 2):
        return None
    assistantparts = examples['accept'].split("<｜Assistant｜>")
    if (len(assistantparts) > 2):
        return None
    examples['prompt'] = parts[0]
    examples['completion'] = parts[1]
    return examples

if __name__ == "__main__":
    print("Starting...")
    
    # Model path
    #model_path = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-1.5B")
    #model_id = "Hankbeasley/Polycrest-Qwen-1.5B"
    model_id = "Hankbeasley/PolycrestSFT-Qwen-1.5B"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # Load model on CPU
    
    from datasets import load_dataset

# Load JSON dataset
    #dataset = load_dataset('json', data_files='path/to/your_dataset.json')

    ds = load_dataset("Hankbeasley/tokenizedAccept")
    ds = ds['train'].filter(lambda x: (len(x['input_ids']) < 7000))
    
    # Create train/test split (80% train, 20% test by default)
    split_dataset = ds.train_test_split(test_size=0.2, shuffle=False)
    split_dataset['train'] = split_dataset['train'].shuffle(seed=42)

    
    trainargs = SFTConfig (
        max_seq_length=7000,
        output_dir="/work/output",
        logging_dir="/work/output/logsr1",           # Directory to save logs
        logging_steps=20,                    # Log every 50 steps
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",         # Evaluate every few steps
        eval_steps=100,                      # Evaluate every 100 steps
        # Save checkpoint every 500 steps
        save_steps=50,
        report_to="tensorboard",
        dataset_kwargs = {
            "skip_prepare_dataset": True,
        },
        push_to_hub=True,
        hub_model_id="Hankbeasley/PolycrestSFT-Qwen-1.5B",
        #push_to_hub_organization="hankbeasley",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable()
    print(model)
    from transformers import DataCollatorWithPadding

    #data_collator = DualDataCollator(tokenizer)
    
    trainer = SFTTrainer(

        model=model,
        processing_class=tokenizer,
        args=trainargs,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        callbacks=[CustomStepCallback(49,"/work/output")]
        #data_collator=data_collator, 
        
        
    )
    trainer.train()