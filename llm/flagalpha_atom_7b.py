# !pip install accelerate bitsandbytes sentencepiece transformers
# It works on T4 on Colab

import torch
import transformers


model_path = "FlagAlpha/Atom-7B"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
).eval()


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
# <s> Human: 介绍一下中国
# </s><s> Assistant: 好的，很高兴为你服务。可以通过以下方式了解更多关于中国的信息：
# 1. 中国官方网站 - 可以访问中华人民共和国政府官网（www.gov.cn）来了解有关中国的最新政策和动态。
# 2. 国际媒体报道 - 通过查看国际主流新闻机构如CNN、BBC等对中国的报道来获得更全面的信息。
# 3. 中国文化知识库 - 推荐一个名为“国家地理中文网”的网站，
# 里面有很多有趣的文章讲解了中国的文化历史以及自然风光等内容。
# 4. 网络社交平台 - 现在有很多华人社群或论坛可以在线交流和学习各种语言和文化的知识，
# 比如微信公众号上的"汉语桥世界华语大会""全球汉学中心联盟"等等。
# 5. 音乐与电影欣赏 - 如果你喜欢音乐和艺术的话，
# 推荐一些经典的歌曲或者纪录片《上甘岭》《英雄儿女》等，
# 它们反映了中国人民在战争年代不畏牺牲的精神品质；
# 另外还可以看看京剧、昆曲这些传统戏曲剧种的表演形式和中国的传统节日习俗表演视频。
# 6. 书籍阅读 - 最后推荐给你几本好书作为参考资料，
# 分别是余秋雨先生的散文集《文化苦旅》、张承志的小说《青春万岁》及鲁迅先生所著的文章合集《朝花夕拾》。
# 希望以上建议对你有所帮助！</s>
