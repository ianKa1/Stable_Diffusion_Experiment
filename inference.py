
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm import tqdm

from peft import LoraConfig, get_peft_model

from peft import PeftModel, PeftConfig

device = "cuda"

model_id = "sd-legacy/stable-diffusion-v1-5"
experiment_name = "sd_lora_bs1_lr1e_4_all_attn_2000"

out_dir = f"{experiment_name}/generate_results"
os.makedirs(out_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
)
pipe.to(device)

pipe.unet = PeftModel.from_pretrained(pipe.unet, "sd_lora_bs1_lr1e_4_all_attn_2000")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

spatial = [
    "a chicken on the left of a car",
    "a person on the left of a cow",
    "a horse on the right of a man",
    "a man on side of a cat",
    "a chicken near a book",
    "a bicycle on the right of a girl",
    "a dog next to a phone",
    "a sheep next to a bicycle",
    "a pig on the bottom of a candle",
    "a butterfly on the left of a phone"
]

for prompt in spatial:
    image = pipe(prompt, num_inference_steps=200).images[0]  
    image.save(f"./{out_dir}/{prompt}.png")