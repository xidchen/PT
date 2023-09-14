# !pip install accelerate bitsandbytes sentencepiece transformers xformers
# It works on T4 on Colab

import torch
import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="baichuan-inc/Baichuan2-13B-Base",
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="baichuan-inc/Baichuan2-13B-Base",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model = model.quantize(4).cuda()


inputs = tokenizer("登鹳雀楼->王之涣\n夜雨寄北->", return_tensors="pt")
inputs = inputs.to("cuda:0")

prediction = model.generate(
    **inputs,
    max_new_tokens=64,
    repetition_penalty=1.1,
)
print(tokenizer.decode(prediction.cpu()[0], skip_special_tokens=True))
