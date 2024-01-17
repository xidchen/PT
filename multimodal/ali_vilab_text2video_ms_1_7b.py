# !pip install accelerate diffusers
# It works on Colab T4

import diffusers
import torch


model_id = "damo-vilab/text-to-video-ms-1.7b"
pipe = diffusers.DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.float16,
)
pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)
pipe.enable_model_cpu_offload()

prompt = ("A house built by a lake in a quiet forest at outskirts of a megacity, "
          "with modern north european decoration and intelligent devices inside. "
          "An elegant lady is lying confortably on a sofa with her feet on the carpet, "
          "sipping coffee, and playing her cellphone.")
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = diffusers.utils.export_to_video(video_frames)
print(video_path)
