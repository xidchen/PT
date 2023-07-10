# !pip install accelerate bitsandbytes transformers
# !nvidia-smi
# It works on A100

import torch
import transformers

# Load the model

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "HuggingFaceH4/starchat-alpha"
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/starchat-alpha",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)


# Code generation and test

def generate_response(input_prompt):
    system_prompt = "<|system|>\nBelow is a conversation between a human user" \
                    " and a helpful AI coding assistant.<|end|>\n"
    user_prompt = f"<|user|>\n{input_prompt}<|end|>\n"
    assistant_prompt = "<|assistant|>"
    full_prompt = system_prompt + user_prompt + assistant_prompt
    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs,
                             eos_token_id=0,
                             pad_token_id=0,
                             max_length=256,
                             early_stopping=True
                             )
    output = tokenizer.decode(outputs[0])
    output = output[len(full_prompt):]
    if "<|end|>" in output:
        cutoff = output.find("<|end|>")
        output = output[:cutoff]
    print(input_prompt + "\n")
    print(output)


generate_response(
    "Implement a recursive function "
    "to calculate the factorial of a given number."
)


# test it
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)


factorial(18)

generate_response(
    "Implement a program that validates "
    "whether the given password satisfies certain criteria, "
    "such as having at least one uppercase letter and one special character."
)

# test it
import re


def validate_password(password):
    # Check length
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    # Check for uppercase letters
    has_uppercase = any(char.isupper() for char in password)
    if not has_uppercase:
        return False, "Password must contain at least one uppercase letter"
    # Check for special characters
    has_special_char = any(re.match(r"[^a-z0-9 ]", char) for char in password)
    if not has_special_char:
        return False, "Password must contain at least one special character"
    # Password meets all criteria
    return True, ""


print("64728 is " + str(validate_password("64728")))
print("64ad3728 is " + str(validate_password("64ad3728")))
print("64Ud3728 is " + str(validate_password("64Ud3728")))
print("64Ud3728R is " + str(validate_password("64Ud3728R")))

generate_response(
    "Create a program that converts temperature from Celsius to Fahrenheit."
)

# test it
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9 / 5) + 32
print("Temperature in Fahrenheit: ", fahrenheit)
