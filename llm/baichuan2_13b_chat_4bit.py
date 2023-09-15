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
# "温故而知新"是一句中国古代名言，出自《论语·为政》篇。这句话的意思是回顾过去，从中汲取经验教训，
# 从而更好地理解现在和未来的发展。具体来说，它鼓励人们在学习和生活中不断地回顾过去的事情，
# 从中发现新的知识和见解，以便更好地应对当前的问题和挑战。
# 这句话强调了回顾过去的重要性，因为我们可以从过去的经验中学习，从而更好地适应不断变化的环境。
# 同时，它也提醒我们要保持开放的心态，对新事物和新观念保持敏感，以便在不断进步的社会中找到自己的定位。
