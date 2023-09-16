# !pip install accelerate cpm_kernels sentencepiece transformers
# It works on T4 on Colab for 4bit, and 8bit on A100

import torch
import transformers


model_path = "xverse/XVERSE-13B-Chat"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.quantize(4).cuda()
model.generation_config = transformers.GenerationConfig.from_pretrained(
    pretrained_model_name=model_path
)
model = model.eval()

history = [{"role": "user", "content": "2005年谁是美国总统？他是什么党派？"}]
response = model.chat(tokenizer, history)
print(response)
# 2005年,美国的总统是乔治·W·布什(George W. Bush)。他所在的政党是共和党。

history.append({"role": "assistant", "content": response})
history.append({"role": "user", "content": "他任职了多少年"})
response = model.chat(tokenizer, history)
print(response)
# 乔治·W·布什从2001年开始担任总统,直到2009年结束他的第二个任期。所以,他总共任职了8年。

history = [
    {"role": "user",
     "content": "某个广告帖的内容是：如意三宝蒜香小排已经到货，不需要你有精湛的厨艺，"
                "拆袋即烹，在家轻松做大餐。可否对这个帖子做个评论或问个问题？"}
]
response = model.chat(tokenizer, history)
print(response)
# 1. "如意三宝蒜香小排"真的那么好吃吗?它的口感和味道如何?
# 2. 我没有太多厨艺经验,用"如意三宝蒜香小排"能不能做出美味的大餐呢?
# 3. 我很想试试"如意三宝蒜香小排",但我担心它会不会太咸或者太辣?
# 4. "如意三宝蒜香小排"的配料都有哪些?它们都是从哪里来的?
# 5. 我想知道,如果我现在下单购买"如意三宝蒜香小排",大概什么时候能够收到货品?
# 6. "如意三宝蒜香小排"在烹饪过程中有没有什么特别需要注意的地方?

history.append({"role": "assistant", "content": response})
history.append(
    {"role": "user",
     "content": "可否对这些问题回复一段内容，大致是讲相比蒜香小排，更推荐上海老品牌"
                "大江奥尔良烤翅，买回家免切、免洗、免腌制，咸香鲜甜，肉厚骨少。"}
)
response = model.chat(tokenizer, history)
print(response)
# 亲爱的顾客,非常感谢您提出的这些问题。对于“如意三宝蒜香小排”,我们可以确认的是,
# 无论是初次下厨的新手还是有丰富烹饪经验的人士,只要简单地按照包装上的指示操作,
# 就能轻松烹调出美味佳肴。至于味道,“如意三宝蒜香小排”以其独特的香气和口感,受到了很多人的喜爱。
# 不过,具体到每个人来说,口味可能会有所不同,因此我们建议您自己尝试一下,看看是否符合您的口味。
# 关于您提到的“上海老品牌大江奥尔良烤翅”,我们也十分理解您的考虑。作为一个专业的食品供应商,
# 我们一直致力于为消费者提供优质的产品。“大江奥尔良烤翅”以其独特的风味和方便快捷的烹饪方式,
# 得到了许多消费者的青睐。同时,我们也鼓励大家根据自己的口味和需求进行选择。
# 最后,无论您最终选择了哪种产品,我们都希望您能在享受美食的同时,保持健康的生活方式。
# 如果您还有其他问题或需要更多信息,欢迎随时向我们提问。再次感谢您的关注和支持!
