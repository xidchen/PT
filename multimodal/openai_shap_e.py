# !pip install accelerate diffusers
# It works on Colab T4

import diffusers


model_id = "openai/shap-e"

pipe = diffusers.ShapEPipeline.from_pretrained(
    pretrained_model_name_or_path=model_id,
).to("cuda")

prompt = "an elegant young lady wearing a black fur outfit in shanghai"
images = pipe(
    prompt,
    guidance_scale=15.0,
    num_inference_steps=64,
).images

gif_path = diffusers.utils.export_to_gif(
    images[0], output_gif_path="lady_3d.gif"
)
