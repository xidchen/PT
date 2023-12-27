# !pip install accelerate datasets transformers
# It works on Colab T4

import datasets
import torch
import transformers


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


model_path = "openai/whisper-large-v3"

model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype,
    low_cpu_mem_usage=True, use_safetensors=True,
)

processor = transformers.AutoProcessor.from_pretrained(model_path)

pipe = transformers.pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

datasets = datasets.load_dataset(
    path="distil-whisper/librispeech_long",
    name="clean",
    split="validation",
)
sample = datasets[0]["audio"]

result = pipe(sample)
print(result["text"])
