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
if __name__ == "__main__":
    print("Starting...")
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2"
    )
    print(tokenizer)
    print(model)