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
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #device_map="auto",  
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2"
    )

print (deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(model,num_gpus_per_node=4) )
print (deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(model,num_gpus_per_node=4) )
     #exit()