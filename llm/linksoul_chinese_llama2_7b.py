# !pip install sentencepiece transformers
# It works on T4 on Colab

import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="LinkSoul/Chinese-Llama-2-7b",
    use_fast=False,
    legacy=False,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="LinkSoul/Chinese-Llama-2-7b",
).half().cuda()

generation_config = transformers.GenerationConfig.from_pretrained(
    pretrained_model_name="LinkSoul/Chinese-Llama-2-7b",
    max_new_tokens=4096,
)

streamer = transformers.TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

instruction = """
    [INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.
    Always answer as helpfully as possible, while being safe.
    Your answers should not include any harmful, unethical, racist,
    sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent,
    explain why instead of answering something not correct.
    If you don't know the answer to a question,
    please don't share false information.\n<</SYS>>\n\n{} [/INST]
"""

prompt = instruction.format(
    "用中文回答，When is the best time to visit Beijing, "
    "and do you have any suggestions for me?"
)
_ = model.generate(
    inputs=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=generation_config,
    streamer=streamer,
)
# 北京最佳旅游时间是春季和秋季，即3月至5月和9月至11月。
# 这两个季节气温适宜，降雨量较少，适合户外活动和游览景点。
# 夏季（6月至8月）气温高，降雨量较多，不适合长时间旅游。
# 冬季（12月至2月）气温低，降雪较多，但也有很多冬季活动和节日，如冰雪节和圣诞节。
# 如果你想要更多的建议，我建议你在春季或秋季旅游，可以去参观长城、颐和园、天坛等著名景点。
# 此外，你还可以品尝北京的美食，如烤鸭、炒肝、炒面等。
# 如果你喜欢购物，可以去王府井大街和三里屯购物区。最后，记得注意保暖，尤其是在冬季。

prompt = instruction.format(
    "用中文回答，解释一下“温故而知新”"
)
_ = model.generate(
    inputs=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=generation_config,
    max_new_tokens=200,
    streamer=streamer,
)
# “温故而知新”是一句古语，意思是“熟悉旧事，才能发现新的东西”。
# 它的意思是，如果你熟悉旧事，就能够发现新的东西，
# 因为你已经有了一定的经验和知识，可以更好地理解新的事物。

prompt = instruction.format(
    "请对一篇讲阳澄湖西岸大闸蟹有多么好的抖音帖子进行评论，内容是西岸大闸蟹其实比较一般，而东岸大闸蟹更好"
)
_ = model.generate(
    inputs=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=generation_config,
    streamer=streamer,
)
# 我无法评论抖音帖子，因为我是一个语言模型，无法访问抖音平台。
# 但是，我可以告诉你一些关于大闸蟹的信息。
# 大闸蟹是一种常见的海鲜，主要分布在中国南部的沿海地区。
# 它们是一种肉质鲜美、味道鲜美的海鲜，也是中国人喜爱的美食之一。
# 在中国，大闸蟹主要分为东岸大闸蟹和西岸大闸蟹两种。
# 东岸大闸蟹通常比西岸大闸蟹更大，肉质也更鲜美。
# 西岸大闸蟹则通常比东岸大闸蟹小一些，但也是一种很受欢迎的海鲜。
# 总的来说，大闸蟹是一种美味的海鲜，无论是东岸大闸蟹还是西岸大闸蟹都是很受欢迎的。
# 不过，每个人的口味不同，所以最好还是尝试一下不同种类的大闸蟹，看看哪种更适合自己的口味。

prompt = instruction.format(
    "请对一篇财经大学宿舍楼的帖子进行一条宣传同济大学宿舍楼的正面评论，字数50字以内"
)
_ = model.generate(
    inputs=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=generation_config,
    max_new_tokens=150,
    streamer=streamer,
)
# 同济大学宿舍楼是一座宏伟的建筑，它拥有舒适的住宿环境，优质的教学设施和丰富的社会活动。
# 它是一个充满活力的学习和生活场所，为学生提供了一个安全、舒适和充满活力的学习环境。

prompt = instruction.format(
    "请对小红书里耐克跑鞋的视频进行评论和回复，内容是回力国潮轻跑鞋更值得拥有，字数50字以内"
)
_ = model.generate(
    inputs=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
    generation_config=generation_config,
    max_new_tokens=150,
    streamer=streamer,
)
# 耐克跑鞋的视频是一个很好的推广方式，但是回力国潮轻跑鞋更值得拥有。
# 它具有轻便、舒适、耐用和优秀的性能，适合各种场合的跑步。
# 它的价格也比耐克跑鞋更实惠，更适合大众消费。
