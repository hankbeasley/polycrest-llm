import shutil
from threading import Thread
import os
import torch
import deepspeed
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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



def preprocess_function2(examples):
    parts = examples['accept'].split('<think>')
    partsrejected = examples['reject'].split('<think>')
    examples['prompt'] = parts[0]
    examples['chosen'] = "<think>" + parts[1]
    examples['rejected'] = "<think>\n" + partsrejected[1]
    return examples

def p(examples):
    examples['text'] = examples['prompt'] +  examples['chosen']
    return examples

if __name__ == "__main__":
    print("Starting...")
    
    # Model path
    #model_path = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-1.5B")
    #model_id = "Hankbeasley/Polycrest-Qwen-1.5B"
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    # Load model on CPU
    
    from datasets import load_dataset

    current = load_dataset("Hankbeasley/polycodertext")['train']
   
    dsinput = current.map(p)
    dsinput = dsinput.filter(lambda x: len(x['text']) < 30000)
    print(dsinput)
    # Create train/test split (80% train, 20% test by default)
    split_dataset = dsinput.train_test_split(test_size=0.1, shuffle=False)
    
    trainargs = SFTConfig (
        max_seq_length=6000,
        output_dir="/work/output",
        logging_dir="/work/output/logsr3",           # Directory to save logs
        logging_steps=20,                    # Log every 50 steps
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="steps",         # Evaluate every few steps
        eval_steps=50,
        dataset_num_proc=16,                      
        save_steps=50,
        report_to="tensorboard",
        bf16=True
        
        # dataset_kwargs = {
        #     "skip_prepare_dataset": True,
        # },
        #push_to_hub=True,
        #hub_model_id="Hankbeasley/PolycrestSFT-Qwen-7B",
        #push_to_hub_organization="hankbeasley",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    print (deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(model,num_gpus_per_node=3) )
    #exit()

    model.gradient_checkpointing_enable()
    print(model)
  
    trainer = SFTTrainer(

        model=model,
        processing_class=tokenizer,
        args=trainargs,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        callbacks=[CustomStepCallback(48,"/work/output")]
        #data_collator=data_collator, 
        
        
    )
    trainer.train()
    trainer.push_to_hub("Hankbeasley/PolycrestSFT-Qwen-7B")