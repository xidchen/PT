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
    "请对一篇讲财经大学宿舍楼的帖子进行一条宣传同济大学宿舍楼的正面评论，直接具体评论，50字左右。",
    history=history
)
print(response)
# 这篇帖子讲述了财经大学宿舍楼的一些故事，让我们对同济大学宿舍楼的设施和环境印象深刻。
# 我们喜欢这里的宽敞明亮、设施齐全，以及便利的地理位置。我们希望能有更多同学关注这里，
# 入住时也能体验到同济大学宿舍楼的舒适和温馨。

_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对某个耐克跑鞋的视频进行评论和回复，内容是回力国潮轻跑鞋更值得拥有，直接具体评论，50字左右。",
    history=history
)
print(response)
# 作为一款具有出色性能和时尚外观的跑鞋，回力国潮轻跑鞋确实值得拥有。
# 它的舒适性和支撑力都能很好地满足跑步时的需求，而且价格也非常亲民，
# 对于喜欢跑步的人来说，它是一个不错的选择。

_, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(
    tokenizer,
    "某个广告帖的内容是：如意三宝蒜香小排已经到货，不需要你有精湛的厨艺，拆袋即烹，"
    "在家轻松做大餐。请直接对该广告帖具体评论，50字左右。",
    history=history
)
print(response)
print()
response, _ = model.chat(
    tokenizer,
    "对这个问题回复一段内容，大致是讲相比蒜香小排，更推荐上海老品牌大江奥尔良烤翅，"
    "买回家免切、免洗、免腌制，咸香鲜甜，肉厚骨少。50字左右。",
    history=history
)
print(response)
# 这个广告帖非常贴心，让人感觉很温馨。虽然不需要有精湛的厨艺，但只要拆袋即烹，就能轻松做大餐，
# 这对于忙碌的现代人来说非常方便。而且，广告中提到的如意三宝蒜香小排，让人很好奇，
# 不知道这是什么味道，有机会一定要尝试一下。
#
# 虽然如意三宝蒜香小排的肉质很好，但是相比起来，我更推荐上海老品牌大江奥尔良烤翅。
# 大江奥尔良烤翅的肉质更加鲜嫩，口感更加丰富。而且，买回家就可以免切、免洗、免腌制，非常方便。
# 无论是烤还是炖，都可以尝到美味的口感，真是一举两得。
