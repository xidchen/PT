# !pip install accelerate bitsandbytes datasets peft sentencepiece transformers

import peft
import torch
import transformers

access_token = "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"

finetune_model_path = "FlagAlpha/Llama2-Chinese-7b-Chat-LoRa"
config = peft.PeftConfig.from_pretrained(finetune_model_path)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, use_fast=False, token=access_token,
)
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    token=access_token,
)
model = peft.PeftModel.from_pretrained(
    model, finetune_model_path, device_map={"": 0}
)
model = model.eval()

input_ids = tokenizer(
    ["<s>Human: 介绍一下北京\n</s><s>Assistant: "],
    return_tensors="pt",
    add_special_tokens=False,
).input_ids.to("cuda")
generate_input = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

generate_ids = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
