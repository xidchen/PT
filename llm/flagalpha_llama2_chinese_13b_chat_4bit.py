# !pip install auto_gptq sentencepiece transformers
# It works on T4 on Colab

import auto_gptq
import transformers


model = auto_gptq.AutoGPTQForCausalLM.from_quantized(
    model_name_or_path="FlagAlpha/Llama2-Chinese-13b-Chat-4bit",
    device="cuda:0",
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="FlagAlpha/Llama2-Chinese-13b-Chat-4bit",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(
    ["<s>Human: 怎么登上火星\n</s><s>Assistant: "],
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
    "pad_token_id": tokenizer.pad_token_id
}
generate_ids = model.generate(**generate_input)

text = tokenizer.decode(generate_ids[0])
print(text)
