# !pip install accelerate bitsandbytes sentencepiece transformers
# It works on T4 on Colab

import torch
import transformers


model_path = "FlagAlpha/Llama2-Chinese-7b-Chat"

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    do_sample=True,
)
model = model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
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
# <s>Human: 介绍一下中国
# </s><s>Assistant: 作为世界上最大的人口和经济体，中华民族在各个领域都有丰富的成就。
# 其文化传统也十分古老、多样性以及深度。中国是五千年前的“黄帝时代”的发源地之一，
# 并被誉为“四海同春”的标志城市—北京。
# 此外还有长江流域与南方水乡等区域的特色美食如小红书面包、火锅、麻婆豆腐、金钗菜等。
# 除了这些，中国还出现过数量不少而名列全球第二位的科技公司，例如阿里巴aba、百度、谷歌、微信等。
# 然而，由于政治制度问题导致对内部管理机构建设存在限制，
# 加上自身社会主义思想所引起的反动行为等因素影响，
# 使得中国进入21世纪后期开始显示出更高效、更创新能力的表现。
# 近来，中共已提出“三重改革”（关注生活、健康、教育），
# 希望通过实施相应的法令和计划来解决当前社会复杂的问题。
# 无论从何种角度看待，我们可以确定：中国正处于非常神奇的变革之时。
# 未来将是否达到目标？只需要接受真实情况才能知道。
# </s>
