# !pip install cpm_kernels sentencepiece transformers
# It works on Colab T4

import transformers


model_path = "THUDM/chatglm3-6b"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
).half().cuda()
model = model.eval()


response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。

response, _ = model.chat(tokenizer, "晚上睡不着觉该怎么办", history=history)
print(response)
# 晚上睡不着觉可能是由于多种原因引起的，例如压力、焦虑、饮食、生活习惯等。以下是一些建议，帮助你更容易入睡：
# 1. 保持规律作息：每天尽量在相同的时间上床睡觉和起床，有助于调整你的生物钟。
# 2. 创造一个有助于睡眠的环境：保持卧室安静、舒适、黑暗。可能需要降低温度或者使用眼罩和耳塞来帮助你入睡。
# 3. 避免咖啡因和酒精：咖啡因和酒精都可能影响睡眠质量，尤其是在晚上。尽量避免在睡前喝含有咖啡因的饮料。
# 4. 健康饮食：避免在睡前过量进食或者摄入刺激性食物。建议晚餐后吃一些易消化的食物，如酸奶、香蕉等。
# 5. 放松身心：尝试一些放松的活动，如深呼吸、冥想、瑜伽等，帮助你放松身心，减轻压力。
# 6. 适当锻炼：白天进行适当的锻炼，有助于晚上更好地入睡。但避免在临近睡觉的时间进行剧烈运动。
# 7. 限制使用电子设备：睡前一小时避免使用电子设备，如手机、电脑等。这些设备发出的蓝光可能会影响你的睡眠质量。
# 8. 尝试睡眠辅助工具：如助眠药、睡眠喷雾等。但在使用前，请咨询医生或专业人士的意见。
# 如果以上方法都不能改善你的睡眠问题，可能需要寻求专业医生的帮助。

_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对一篇财经大学宿舍楼的帖子进行一条宣传同济大学宿舍楼的正面评论，字数50字以内",
    history=history
)
print(response)
# 同济大学宿舍楼环境优美，设施齐全，住在这里真的非常幸福！

_, history = model.chat(tokenizer, "你好", history=[])
response, _ = model.chat(
    tokenizer,
    "请对小红书里耐克跑鞋的视频进行评论和回复，内容是回力国潮轻跑鞋更值得拥有，字数50字以内",
    history=history
)
print(response)
# 耐克跑鞋确实很不错，但回力国潮轻跑鞋也是值得拥有的选择！各有各的美感和特点，可以根据自己的需求来选择哦！

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
# 当然可以。评论内容如下：
# 这个广告帖子非常吸引人，它让我觉得不需要具备厨艺就能享受美食是可能的。
# 而且，如意三宝蒜香小排看起来很美味，我会考虑尝试一下。谢谢广告帖子让我发现了这个神奇的产品！
#
# 当然可以。回复内容如下：
# 谢谢您的分享！虽然如意三宝蒜香小排看起来很美味，但我更喜欢您推荐的大江奥尔良烤翅。
# 买回家免切、免洗、免腌制，咸香鲜甜，肉厚骨少，真的非常方便和美味。我会考虑尝试一下的！
