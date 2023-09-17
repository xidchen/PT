# !pip install sentencepiece transformers
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
    device="cuda",
)


response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。

response, _ = model.chat(tokenizer, "晚上睡不着觉该怎么办", history=history)
print(response)
# 以下是一些有助于改善睡眠质量的建议：
# 1. 晚上避免过度兴奋：在睡觉前 1-2 小时内避免看电视、打电子游戏或进行其他带有强烈刺激的活动。
# 2. 规律作息：尽量在相同的时间上床睡觉和起床，形成一个稳定的睡眠节律。
# 3. 创建舒适的睡眠环境：保持卧室安静、凉爽、黑暗、舒适，可尝试使用睡眠面罩、耳塞或舒适的枕头。
# 4. 放松身心：在睡觉前进行一些轻松的活动，如阅读、听轻柔的音乐或洗个热水澡，有助于缓解压力和焦虑。
# 5. 限制躺在床上的时间：如果在床上躺了 20 分钟仍然无法入睡，不要继续躺在床上，
# 而是起床做些轻松的活动，直到感到困倦再返回床上。
# 6. 饮食注意事项：避免在睡觉前进食大量咖啡因、酒精、烟草等刺激性食物和饮料。
# 7. 锻炼身体：适当的身体锻炼可以帮助改善睡眠质量，但尽量避免在睡觉前进行剧烈运动。
# 如果这些方法仍然无法帮助你改善睡眠质量，请及时咨询医生。

_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对一篇财经大学宿舍楼的帖子进行一条宣传同济大学宿舍楼的正面评论，字数50字以内",
    history=history
)
print(response)
# 这篇帖子宣传同济大学宿舍楼的正面评论如下：
# 同济大学宿舍楼楼内设施齐全，环境舒适，安全美观，周边生活设施完善，交通便利，靠近地铁站，
# 学术氛围浓厚，是 studentspace 中的优秀代表。

_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对小红书里耐克跑鞋的视频进行评论和回复，内容是回力国潮轻跑鞋更值得拥有，字数50字以内",
    history=history
)
print(response)
# "回力国潮轻跑鞋更值得拥有，时尚舒适，轻盈透气，跑步必备！"

_, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(
    tokenizer,
    "某个广告帖的内容是：如意三宝蒜香小排已经到货，不需要你有精湛的厨艺，拆袋即烹，"
    "在家轻松做大餐。可否在这个帖子下面做个评论？只给出评论内容。",
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
# 这个帖子真的太实用了！特别是对于像我这样的懒人来说，更是如此。而且味道也很好，我相信一定会很好吃！
#
# 相比蒜香小排，我更推荐上海老品牌大江奥尔良烤翅。它的口感更加鲜香扑鼻，肉厚骨少，
# 而且买回家就可以免切、免洗、免腌制，非常方便。烤出来的味道咸香鲜甜，非常适合家庭聚餐或烤肉派对。
