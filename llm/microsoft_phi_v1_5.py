# !pip install einops transformers
# It works on T4 on Colab

import torch
import transformers

torch.set_default_device("cuda")


model_path = "microsoft/phi-1_5"

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
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
#     ```python
#     def print_prime(n):
#         """
#         Print all primes between 1 and n
#         """
#
#         primes = []
#         for num in range(2, n+1):
#             is_prime = True
#             for i in range(2, int(math.sqrt(num))+1):
#                 if num % i == 0:
#                     is_prime = False
#                     break
#             if is_prime:
#                 primes.append(num)
#         print(primes)
#     ```
