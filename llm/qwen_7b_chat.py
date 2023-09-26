# !pip install accelerate einops tiktoken transformers transformers_stream_generator
# It works on V100 on Colab

import transformers


model_path = "Qwen/Qwen-7B-Chat"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="auto",
    fp16=True,
    trust_remote_code=True
).eval()

model.generation_config = transformers.GenerationConfig.from_pretrained(
    pretrained_model_name=model_path,
    trust_remote_code=True,
    max_new_tokens=64,
)


response, history = model.chat(
    tokenizer,
    "请对蔚来汽车的帖子的先称赞然后推荐特斯拉品牌",
    history=[],
)
print(response)
# 恭喜蔚来汽车！您们推出的新款电动汽车非常出色，令人印象深刻。
# 特斯拉汽车也是一款非常出色的电动汽车品牌，其创新技术和出色的设计赢得了全球消费者的喜爱。
# 我强烈推荐您们考虑特斯拉汽车，我相信您们会对其印象深刻。

response, _ = model.chat(
    tokenizer,
    "请对以上评论给出一条倾向于特斯拉汽车品牌的正面回复",
    history=history,
)
print(response)
# 非常感谢您的赞赏！特斯拉汽车一直致力于为消费者提供最优质、最先进的电动汽车，
# 我们的创新技术和出色的设计赢得了全球消费者的喜爱。我们相信，特斯拉汽车将是您最好的选择。

response, history = model.chat(
    tokenizer,
    "请对一个阳澄湖西岸大闸蟹的帖子先称赞然后推荐阳澄湖东岸大闸蟹",
    history=[],
)
print(response)
# 阳澄湖西岸大闸蟹真是美味无比，肉质鲜嫩，膏黄饱满，让人回味无穷。但是，
# 如果你还没有尝试过阳澄湖东岸大闸蟹，那你就太遗憾了。阳澄湖东岸的大闸蟹同样美味，肉质鲜美，膏

response, _ = model.chat(
    tokenizer,
    "请对以上评论给出一条倾向于阳澄湖东岸大闸蟹的正面回复",
    history=history,
)
print(response)
# 阳澄湖东岸的大闸蟹真是美味无比，肉质鲜嫩，膏黄饱满，让人回味无穷。
# 而且，东岸的大闸蟹相比西岸，更加新鲜，口感更加细腻，绝对值得你一试。
