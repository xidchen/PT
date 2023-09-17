# !pip install cpm_kernels sentencepiece transformers
# It works on T4 on Colab

import transformers


model_path = "THUDM/chatglm2-6b"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
).quantize(4).cuda()


_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对一篇财经大学宿舍楼的帖子进行一条宣传同济大学宿舍楼的正面评论，直接给出评论，50字左右。",
    history=history
)
print(response)
# 这篇财经大学宿舍楼的帖子让我想起同济大学宿舍楼，他们的宿舍楼环境优美，设施齐全，非常羡慕。

_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对某个耐克跑鞋的视频进行评论和回复，内容是回力国潮轻跑鞋更值得拥有，50字左右，不要说很抱歉无法评论的话",
    history=history
)
print(response)
# 作为一款具有出色性能和时尚外观的跑鞋，回力国潮轻跑鞋确实值得拥有。
# 它的舒适性和支撑力都能满足跑步时的需求，而且还能提供足够的灵活性和自由度。
# 对于喜欢跑步的人来说，这款鞋是一个不错的选择。

_, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(
    tokenizer,
    "某个广告帖的内容是：如意三宝蒜香小排已经到货，不需要你有精湛的厨艺，拆袋即烹，"
    "在家轻松做大餐。可否在这个广告帖下面做个评论？只给出评论内容。",
    history=history
)
print(response)
print()
response, _ = model.chat(
    tokenizer,
    "对这个问题回复一段内容，大致是讲相比蒜香小排，更推荐上海老品牌大江奥尔良烤翅，"
    "买回家免切、免洗、免腌制，咸香鲜甜，肉厚骨少。只给出回复内容。",
    history=history
)
print(response)
# 味道好，服务好，价格低，值得购买。
#
# 虽然蒜香小排已经很好吃，但我更推荐上海老品牌大江奥尔良烤翅。它的肉厚骨少，咸香鲜甜，
# 买回家免切、免洗、免腌制，更加方便快捷。
