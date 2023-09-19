# !pip install sentencepiece swissarmytransformer transformers
# It works on V100 on Colab

import transformers


model_path = "THUDM/visualglm-6b"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
)

model = transformers.AutoModel.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
).half().cuda()


image_path = "/content/5.jpg"
response, history = model.chat(
    tokenizer,
    image_path,
    "描述这张图片。",
    history=[]
)
print(response)
response, _ = model.chat(
    tokenizer,
    image_path,
    "这狗是什么品种？",
    history=history)
print(response)
# 照片中的男子戴着灰色帽子和太阳镜，穿着黑色夹克，正站在狗的面前。
# 他看着狗狗，微笑着，似乎心情愉悦。他的手里拿着手机拍照，记录下这一刻的美好时刻。
# 或许这是他与宠物之间的一次互动，也可能是他在回忆过去的时候，或者是在思考未来的时刻。
# 无论如何，这个场景都充满了温馨、幸福和爱的味道。
#
# 这张照片中并没有明确说明这只狗的品种，但根据男子的服装以及狗的姿态来看，
# 可以猜测是一只白色的大型犬或拉布拉多犬。
