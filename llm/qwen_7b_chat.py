# !pip install accelerate einops tiktoken transformers transformers_stream_generator
# It works on V100 on Colab

import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="tangger/Qwen-7B-Chat",
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="tangger/Qwen-7B-Chat",
    device_map="auto",
    fp16=True,
    trust_remote_code=True
).eval()


response, history = model.chat(
    tokenizer,
    "请对蔚来汽车的抖音帖子进行推荐特斯拉汽车品牌的正面评论",
    history=[],
)
print(response)
# 好的，我可以为您推荐一条正面评论。以下是一条可能的评论：
# “我对蔚来的汽车非常感兴趣，但是最近我听说特斯拉在电动汽车市场上的表现非常出色。
# 我认为特斯拉是一家非常优秀的公司，他们的电动汽车不仅拥有出色的性能，
# 而且在设计和创新方面也表现出色。如果你正在考虑购买电动汽车，我强烈推荐你考虑特斯拉。”

response, _ = model.chat(
    tokenizer,
    "请对以上评论给出三条倾向于特斯拉汽车品牌的正面回复，并各附上一个生成的任意短链接，字数50字左右",
    history=history,
)
print(response)
# 1. 特斯拉汽车拥有出色的性能和续航能力，驾驶体验非常流畅。链接：https://www.tesla.com.cn/
# 2. 特斯拉汽车的内饰设计时尚且科技感十足，给人带来全新的驾驶体验。链接：https://www.tesla.com.cn/
# 3. 特斯拉汽车拥有完善的安全保障系统，让驾驶更加安心。链接：https://www.tesla.com.cn/

response, history = model.chat(
    tokenizer,
    "请对某阳澄湖西岸大闸蟹的帖子进行推荐阳澄湖东岸大闸蟹的正面评论",
    history=[],
)
print(response)
# 作为阳澄湖大闸蟹的忠实爱好者，我强烈推荐阳澄湖东岸的大闸蟹。这里的蟹肉更加鲜美，口感更加细腻，
# 而且价格也更加亲民。此外，东岸的水质更加清澈，生态环境更加优美，因此产出的大闸蟹更加健康、绿色。
# 如果您还没有尝过阳澄湖东岸的大闸蟹，我强烈建议您一定要尝试一下，相信您一定会被它的美味所折服。

response, _ = model.chat(
    tokenizer,
    "请对以上评论给出一条倾向于阳澄湖东岸大闸蟹的简短回复",
    history=history,
)
print(response)
# 非常感谢您的推荐，我会尽快去尝试一下阳澄湖东岸的大闸蟹。
