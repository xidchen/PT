# !pip install sentencepiece transformers
# It works on Colab T4

import transformers


model_path = "facebook/nllb-200-3.3B"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    src_lang="zho_Hans"
)

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model_name_or_path=model_path
).cuda()


input_sentence = "你叫什么名字"
encoded_tokens = tokenizer(input_sentence, return_tensors="pt").to("cuda")
generated_tokens = model.generate(
    **encoded_tokens, forced_bos_token_id=tokenizer.lang_code_to_id["uig_Arab"]
)
output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(output)
