import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from torchvision import transforms
from peft import PeftModel, PeftConfig
from diffusers import DDIMScheduler


experiment_name = "sd_lora_bs1_lr1e_5_attn_2000_v2"

model_id = "sd-legacy/stable-diffusion-v1-5"

device = "mps" if torch.backends.mps.is_available() else "cuda"
print("Using device:", device)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe.unet = PeftModel.from_pretrained(pipe.unet, f"{experiment_name}/checkpoint-400")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.to(device)

file_path = '../T2I-CompBench/examples/dataset/spatial_val.txt'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        sample_list = [line.strip() for line in f.readlines() if line.strip()]
    
    sample_list = sample_list[0:100]

    print(f"Extracted {len(sample_list)} lines.")
    print("First 5 items:", sample_list[:5])
else:
    print(f"File '{file_path}' not found.")

output_dir = f"./T2I-CompBench/{experiment_name}/samples"
os.makedirs(output_dir, exist_ok=True)
count = 0

for prompt in sample_list:
    image = pipe(prompt).images[0]
    if count < 10:
      image.save(f"{output_dir}/{prompt}_00000{count}.png")
    if count >= 10 and count < 100:
      image.save(f"{output_dir}/{prompt}_0000{count}.png")
    if count >= 100 and count < 1000:
      image.save(f"{output_dir}/{prompt}_000{count}.png")
    count += 1