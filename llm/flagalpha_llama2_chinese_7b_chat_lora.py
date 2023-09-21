# !pip install accelerate bitsandbytes datasets peft sentencepiece transformers
# It works on T4 on Colab

import peft
import torch
import transformers


access_token = "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"

finetune_model_path = "FlagAlpha/Llama2-Chinese-7b-Chat-LoRa"
config = peft.PeftConfig.from_pretrained(finetune_model_path)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.base_model_name_or_path, use_fast=False, token=access_token,
)
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    token=access_token,
)
model = peft.PeftModel.from_pretrained(
    model, finetune_model_path, device_map={"": 0}
)
model = model.eval()


input_ids = tokenizer(
    ["<s>Human: 介绍一下北京\n</s><s>Assistant: "],
    return_tensors="pt",
    add_special_tokens=False,
).input_ids.to("cuda")
generate_input = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
}

generate_ids = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
# <s>Human: 介绍一下北京
# </s><s>Assistant: 当然，我可以为您介绍北京的几个关键点。首先是中国最高政治机构——人民大会堂，
# 建于1950年代初期，位于东城区天安门广场西侧，由张自忠设计；其次就是长江路上的故宫博物院，
# 这座古老而著名的皇帝府馆在明清两朝时间内被用作居所和行政中心，现已成为全球知名文化保护项目之一;
# 第三、四分别是二十里店及周边景色与地标性建筑如前进公司等，该处也称“小巴麟山”，因此得名，
# 有美丽的花草水石画面，同样值得参考学习；五六则是原来的工商业街道-新华街，
# 主要包括了电影、音乐、书法、茶器、手工制品、服装、家居等多种类型的特色市集，
# 非常符合消费者对生活方式的需求。希望能给到你们更好的体验！请问您还想查看其他部分？
# </s>
