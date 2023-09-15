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
# 登鹳雀楼->王之涣
# 夜雨寄北->李商隐
# 望岳->杜甫
# 春望->杜甫
# 石壕吏->杜甫
# 茅屋为秋风所破歌->杜甫
# 白雪歌送武判官归京->岑参
# 早春呈水部张十八员外->韩愈
# 酬乐天扬州初逢
