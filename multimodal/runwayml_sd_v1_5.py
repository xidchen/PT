# !pip install accelerate diffusers
# It works on Colab T4

import diffusers
import torch


model_id = "runwayml/stable-diffusion-v1-5"
pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = ("A house built by a lake in a quiet forest at outskirts of a megacity, "
          "with modern north european decoration and intelligent devices inside. "
          "An elegant lady is lying confortably on a sofa with her feet on the carpet, "
          "sipping coffee, and playing her cellphone.")
image = pipe(prompt).images[0]
image.save("generated_image.png")
