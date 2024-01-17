# !pip install accelerate apex bitsandbytes einops sentencepiece
# !pip install timm torch==2.1.0 transformers xformers
# It works on Colab T4

import PIL.Image
import torch
import transformers


MODEL_PATH = "THUDM/cogagent-chat-hf"
TOKENIZER_PATH = "lmsys/vicuna-7b-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=TOKENIZER_PATH,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
)

text_only_template = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: {} ASSISTANT: "
)

while True:
    image_path = input("image path >>>>> ")
    text_only_first_query = True
    if image_path == "":
        print("You did not enter image path, the following will be a plain text conversation.")
        image = None
        text_only_first_query = True
    else:
        image = PIL.Image.open(image_path).convert('RGB')

    history = []

    while True:
        query = input("Human: ")
        if query == "clear":
            break

        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = ''
                for _, (old_query, response) in enumerate(history):
                    old_prompt += old_query + " " + response + "\n"
                query = old_prompt + "USER: {} ASSISTANT:".format(query)

        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, template_version='base'
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, images=[image]
            )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch.float16)]]
            if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [
                [input_by_model['cross_images'][0].to(DEVICE).to(torch.float16)]
            ]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048, "do_sample": False}  # "temperature": 0.9
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
        history.append((query, response))
