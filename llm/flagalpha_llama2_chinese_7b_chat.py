# !pip install accelerate bitsandbytes sentencepiece transformers

import torch
import transformers


model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="FlagAlpha/Llama2-Chinese-7b-Chat",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    do_sample=True,
)
model = model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="FlagAlpha/Llama2-Chinese-7b-Chat",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(
    ["<s>Human: 介绍一下中国\n</s><s>Assistant: "],
    return_tensors="pt",
    add_special_tokens=False,
).input_ids.to(model.device)

generate_input = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "bos_token_id": tokenizer.bos_token_id,
}

generated_ids = model.generate(**generate_input)
text = tokenizer.decode(generated_ids[0])
print(text)
