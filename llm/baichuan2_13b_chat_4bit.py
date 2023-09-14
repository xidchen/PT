# !pip install accelerate bitsandbytes sentencepiece transformers xformers
# It works on T4 on Colab

import torch
import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="baichuan-inc/Baichuan2-13B-Chat",
    use_fast=False,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="baichuan-inc/Baichuan2-13B-Chat",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = model.quantize(4).cuda()
model.generation_config = transformers.GenerationConfig.from_pretrained(
    pretrained_model_name="baichuan-inc/Baichuan2-13B-Chat",
)


messages = [{"role": "user", "content": "解释一下“温故而知新”"}]
response = model.chat(tokenizer, messages)
print(response)
