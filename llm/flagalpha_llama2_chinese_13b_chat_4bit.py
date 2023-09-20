# !pip install auto_gptq sentencepiece transformers
# It works on T4 on Colab

import auto_gptq
import transformers


model_path = "FlagAlpha/Llama2-Chinese-13b-Chat-4bit"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

model = auto_gptq.AutoGPTQForCausalLM.from_quantized(
    model_name_or_path=model_path,
    device="cuda:0",
)


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
# <s>Human: 怎么登上火星
# </s><s>Assistant: 目前，人类还没有实现登陆火星的能力。
# 许多国家和机构已经开始了对火星进行研究和发展计划，
# 但是这些都需要大量时间、金额以及技术支持才可成功完成。
# 如果你想知道更加细节信息，建议查看相关专业网站或者参与一些科学活动来获取最新消息。
# </s>
