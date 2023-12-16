# !pip install einops
# It works on Colab T4

import torch
import transformers

torch.set_default_device("cuda")


model_path = "microsoft/phi-2"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
    torch_dtype="auto",
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
    torch_dtype="auto",
)

inputs = tokenizer(
    '''
    ```python
    def print_prime(n):
        """
        Print all primes between 1 and n
        """
    ''',
    return_tensors="pt",
    return_attention_mask=False,
)
outputs = model.generate(**inputs, max_new_tokens=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
# ```python
# def print_prime(n):
#     """
#     Print all primes between 1 and n
#     """
#
#     for i in range(2, n+1):
#         if is_prime(i):
#             print(i)
#
# print_prime(20)
# ```
