from threading import Thread
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from transformers import DataCollatorWithPadding
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase
class DualDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a pad token defined.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Get all sequence lengths for both branches
        chosen_lens = [len(feature["input_ids_chosen"]) for feature in features]
        rejected_lens = [len(feature["input_ids_rejected"]) for feature in features]
        # Compute the common max length for the batch
        common_max_length = max(max(chosen_lens), max(rejected_lens))

        def pad_sequence(seq, pad_token_id):
            return seq + [pad_token_id] * (common_max_length - len(seq))

        batch_input_ids_chosen = []
        batch_attention_mask_chosen = []
        batch_input_ids_rejected = []
        batch_attention_mask_rejected = []

        for feature in features:
            batch_input_ids_chosen.append(
                pad_sequence(feature["input_ids_chosen"], self.tokenizer.pad_token_id)
            )
            batch_attention_mask_chosen.append(
                pad_sequence(feature["attention_mask_chosen"], 0)
            )
            batch_input_ids_rejected.append(
                pad_sequence(feature["input_ids_rejected"], self.tokenizer.pad_token_id)
            )
            batch_attention_mask_rejected.append(
                pad_sequence(feature["attention_mask_rejected"], 0)
            )

        return {
            "input_ids_chosen": torch.tensor(batch_input_ids_chosen, dtype=torch.long),
            "attention_mask_chosen": torch.tensor(batch_attention_mask_chosen, dtype=torch.long),
            "input_ids_rejected": torch.tensor(batch_input_ids_rejected, dtype=torch.long),
            "attention_mask_rejected": torch.tensor(batch_attention_mask_rejected, dtype=torch.long),
        }

def find_max_length(dataset, column_name):
    return max(len(tokens) for tokens in dataset[column_name])
def preprocess_function(examples):
    # Tokenize and truncate the `chosen` column
    chosen_tokens = tokenizer(
        examples["chosen"],
        truncation=True,
        max_length=8000,  # Set to model's maximum token length
        #padding="max_length",  # Ensure consistent sequence lengths
        #return_tensors="pt",
    )
    
    # Tokenize and truncate the `rejected` column
    rejected_tokens = tokenizer(
        examples["rejected"],
        truncation=True,
        max_length=8000,
       # padding="max_length",
        #return_tensors="pt",
    )
    
    # # Add required pretokenized columns
    # examples["input_ids_chosen"] = chosen_tokens["input_ids"].tolist()
    # examples["attention_mask_chosen"] = chosen_tokens["attention_mask"].tolist()
    # examples["input_ids_rejected"] = rejected_tokens["input_ids"].tolist()
    # examples["attention_mask_rejected"] = rejected_tokens["attention_mask"].tolist()
    if len(chosen_tokens["input_ids"]) > 5000 or len(rejected_tokens["input_ids"]) > 5000:
        return None 
    examples["input_ids_chosen"] = chosen_tokens["input_ids"]
    examples["attention_mask_chosen"] = chosen_tokens["attention_mask"]
    examples["input_ids_rejected"] = rejected_tokens["input_ids"]
    examples["attention_mask_rejected"] = rejected_tokens["attention_mask"]
    if "testname" in examples:
        del examples["testname"]
    return examples

if __name__ == "__main__":
    print("Starting...")

    


    

    
    # Model path
    #model_path = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-1.5B")
    #model_id = "Hankbeasley/Polycrest-Qwen-1.5B"
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # Load model on CPU
    
    from datasets import load_dataset

# Load JSON dataset
    #dataset = load_dataset('json', data_files='path/to/your_dataset.json')

    ds = load_dataset("Hankbeasley/polycoder")
    ds = ds.rename_column("accept", "chosen")
    ds = ds.rename_column("reject", "rejected")

    ds = ds.map(preprocess_function, batched=True)
    
    ds = ds.remove_columns("chosen")
    ds = ds.remove_columns("rejected")
    def filter_long_examples(example):
        max_length = 6500
        return len(example["input_ids_chosen"]) <= max_length and len(example["input_ids_rejected"]) <= max_length

    ds = ds.filter(filter_long_examples)
    

    full_dataset = ds['train']

    # Create train/test split (80% train, 20% test by default)
    split_dataset = full_dataset.train_test_split(test_size=0.2)

    print(split_dataset)
    max_length_chosen = find_max_length(split_dataset["train"], "input_ids_chosen")
    max_length_rejected = find_max_length(split_dataset["train"], "input_ids_rejected")
    print(max_length_chosen)
    print(max_length_rejected)
    print(min(len(tokens) for tokens in split_dataset["train"]["input_ids_rejected"]))
    trainargs = RewardConfig (
        
        output_dir="output",
        logging_dir="output/logs",           # Directory to save logs
        logging_steps=50,                    # Log every 50 steps
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        #evaluation_strategy="steps",         # Evaluate every few steps
        #eval_steps=100,                      # Evaluate every 100 steps
        save_steps=500,                      # Save checkpoint every 500 steps
        report_to="tensorboard",
        push_to_hub=True,
        push_to_hub_model_id="Polycrest-Qwen-1.5B",
        
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

    data_collator = DualDataCollator(tokenizer)
    
    trainer = RewardTrainer(

        model=model,
        processing_class=tokenizer,
        args=trainargs,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        data_collator=data_collator, 
        
        
    )
    trainer.train()



    # # Input prompt
    # prompt = "Hello, how are you today?"
    # inputs = tokenizer(
    #     prompt,
    #     return_tensors="pt",          # Return PyTorch tensors
    #     padding=True,                 # Add padding to match model input requirements
    #     truncation=True,              # Truncate if the input is too long
    #     max_length=50,                # Set a maximum input length
    # )

    # # Initialize the streamer
    # streamer = TextIteratorStreamer(
    #     tokenizer,
    #     skip_special_tokens=True,  # Skip special tokens in the output
    # )

    # # Launch the generation in a separate thread to allow real-time streaming
    # def generate_text():
    #     model.generate(
    #         input_ids=inputs.input_ids,
    #         attention_mask=inputs.attention_mask,
    #         max_length=100,
    #         temperature=0.7,
    #         pad_token_id=tokenizer.eos_token_id,
    #         streamer=streamer,  # Pass the streamer
    #     )

    # generation_thread = Thread(target=generate_text)
    # generation_thread.start()

    # # Stream and print the generated tokens
    # print("Generated Text (streaming):", end=" ", flush=True)
    # for token in streamer:
    #     print(token, end="", flush=True)
    # print("\nDone!")
