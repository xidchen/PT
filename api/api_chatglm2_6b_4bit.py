# !pip install cpm_kernels fastapi nest_asyncio sentencepiece transformers uvicorn
# It works on T4 on Colab

import datetime
import fastapi
import json
import nest_asyncio
import torch
import transformers
import uvicorn


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


nest_asyncio.apply()

app = fastapi.FastAPI()


@app.post("/")
async def create_item(request: fastapi.Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get("prompt")
    history = json_post_list.get("history")
    max_length = json_post_list.get("max_length")
    top_p = json_post_list.get("top_p")
    temperature = json_post_list.get("temperature")
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        max_length=max_length if max_length else 2048,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.05,
    )
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%m:%s")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time,
    }
    print(f"[{time}], prompt: \"{prompt}\", response: \"{response}\"")
    torch_gc()
    return answer


if __name__ == "__main__":
    model_path = "THUDM/chatglm2-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
    ).quantize(4).cuda().eval()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
