# !pip install sentencepiece transformers
# It works on Colab T4

import transformers


model_path = "facebook/m2m100_1.2B"

tokenizer = transformers.M2M100Tokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path
)

model = transformers.M2M100ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_path
).cuda()


tokenizer.src_lang = "zh"
input_text = "你叫什么名字"
encoded_tokens = tokenizer(input_text, return_tensors="pt").to("cuda")
generated_tokens = model.generate(
    **encoded_tokens, forced_bos_token_id=tokenizer.get_lang_id("fr")
)
output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(output)
