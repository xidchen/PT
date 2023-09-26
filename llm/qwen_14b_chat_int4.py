# !pip install accelerate auto-gptq einops optimum tiktoken
# !pip install transformers transformers_stream_generator
# It works on T4 on Colab

import transformers


model_path = "Qwen/Qwen-14B-Chat-Int4"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    device_map="auto",
    trust_remote_code=True,
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
# 蔚来汽车的帖子非常出色，他们展示了自己的创新精神和对高品质产品的承诺。
# 他们的电动汽车不仅具有出色的性能，而且在设计和功能上也非常出色。
# 然而，我也想推荐特斯拉品牌。特斯拉是电动汽车市场的领导者，
# 他们的电动汽车在性能、续航里程和创新技术方面都表现出色。

response, _ = model.chat(
    tokenizer,
    "请对以上评论给出一条倾向于特斯拉汽车品牌的正面回复",
    history=history,
)
print(response)
# 非常感谢您的评论，我们非常认同特斯拉在电动汽车市场的领导地位。
# 特斯拉的电动汽车不仅在性能和续航里程方面表现出色，而且他们的创新技术也引领了整个行业的发展。
# 我们非常期待特斯拉未来的发展，以及他们能够为我们带来更多的惊喜。

response, history = model.chat(
    tokenizer,
    "请对一个阳澄湖西岸大闸蟹的帖子先称赞然后推荐阳澄湖东岸大闸蟹",
    history=[],
)
print(response)
# 阳澄湖西岸大闸蟹，真是美味无比！蟹肉鲜美，口感细腻，让人回味无穷。
# 特别是阳澄湖的水质清澈，使得大闸蟹的肉质更加鲜美。
# 如果你还没有尝试过阳澄湖大闸蟹，我强烈推荐你一定要去品尝一下

response, _ = model.chat(
    tokenizer,
    "请对以上评论给出一条倾向于阳澄湖东岸大闸蟹的正面回复",
    history=history,
)
print(response)
# 阳澄湖东岸大闸蟹也是不容错过的美味！这里的水质同样清澈，而且东岸的大闸蟹肉质更加鲜美，
# 口感更加细腻。如果你想要尝试一种不同的口感，我强烈推荐你去品尝一下阳澄湖东岸大闸蟹。
