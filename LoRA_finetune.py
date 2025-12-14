import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm import tqdm

from peft import LoraConfig, get_peft_model


# ============================================
# 1. Config
# ============================================
model_id = "sd-legacy/stable-diffusion-v1-5"
dataset_dir = "./vsr_sd_full"      
lora_rank = 4
train_steps = 2000
learning_rate = 1e-4
batch_size = 1
resolution = 512
experiment_name = "sd_lora_bs1_lr1e_5_all_attn_2000"
output_dir = f"./{experiment_name}" #MODIFY

device = "mps" if torch.backends.mps.is_available() else "cuda"
print("Using device:", device)


# ============================================
# 2. Load model (Stable Diffusion v1.5)
# ============================================
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
)
pipe.to(device)

lora_module_list = []

# MODIFY HERE
# for name, module in pipe.unet.named_modules():
#     if 'resnet' in name and 'up_blocks' in name and 'conv' in name and not 'conv_shortcut' in name:
#         print(name)
#         lora_module_list.append(name)

lora_module_list = ['to_q','to_v','to_k','to_out.0']
print(lora_module_list)

# Freeze base model
pipe.unet.requires_grad_(False)

# ============================================
# 3. Apply LoRA to UNet cross-attention layers
# ============================================
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,
    target_modules=lora_module_list, #MODIFY
    lora_dropout=0.0,
    bias="none",
)

pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.print_trainable_parameters()

print(pipe.unet)

# ============================================
# 4. Dataset
# ============================================
class CustomDataset(Dataset):
    def __init__(self, root):
        self.image_dir = os.path.join(root, "images")
        self.caption_file = os.path.join(root, "captions.txt")

        with open(self.caption_file, "r") as f:
            lines = f.read().splitlines()

        self.data = []
        for line in lines:
            filename, caption = line.split("\t")
            self.data.append((filename, caption))

        self.preprocess = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, caption = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        image = self.preprocess(image)
        return image, caption


dataset = CustomDataset(dataset_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ============================================
# 5. Optimizer + Scheduler
# ============================================
optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=learning_rate)

noise_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=train_steps,
)

import json
os.makedirs(output_dir, exist_ok=True)
lora_config.save_pretrained(output_dir)

train_config = {
    "lora_rank": 4,
    "train_steps": train_steps,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "resolution": resolution
}

with open(f"./{output_dir}/train_config.json", "w") as f:
    json.dump(train_config, f, indent=2)

import wandb
wandb.init(
    project="stable-diffusion-training",   # change this
    name=f"{experiment_name}_2",           # optional run name
    config={
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "resolution": resolution,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": lr_scheduler.__class__.__name__,
        "model": "sd-legacy/stable-diffusion-v1-5",
        "train_unet": True,
        "train_text_encoder": False,
    },
    reinit=True
)

EVAL_PROMPTS = [
    "a dog next to a phone",
    "a sheep next to a bicycle",
    "a pig on the bottom of a candle",
    "a butterfly on the left of a phone"
]

EVAL_SEED = 42
EVAL_STEPS = 50

def save_samples(pipe, step):
    out_dir = f"{output_dir}/train_samples"
    os.makedirs(out_dir, exist_ok=True)

    generator = torch.Generator(device=pipe.device).manual_seed(EVAL_SEED)

    images = pipe(
        EVAL_PROMPTS,
        num_inference_steps=EVAL_STEPS,
        generator=generator,
    ).images

    for i, img in enumerate(images):
        img.save(f"{out_dir}/step_{step:06d}_{EVAL_PROMPTS[i]}.png")


# ============================================
# 6. Training Loop
# ============================================
pipe.text_encoder.requires_grad_(False)
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

pipe.unet.train()

global_step = 0

for epoch in range(100):  # loop until reaching steps
    for batch in dataloader:
        if global_step >= train_steps:
            break

        images, captions = batch
        images = images.to(device)

        # Encode text
        inputs = tokenizer(
            list(captions),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Add noise
        with torch.no_grad():
            latents = pipe.vae.encode(images).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()

        # 2. Sample noise
        noise = torch.randn_like(latents)

        # 3. Add noise according to timestep
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        global_step += 1

        wandb.log(
            {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            },
            step=global_step,
        )

        if global_step % 50 == 0:
            print(f"Step {global_step} / {train_steps}, Loss = {loss.item():.4f}")
        if global_step % 50 == 0:
            pipe.unet.eval()
            with torch.no_grad():
                save_samples(pipe, global_step)
            pipe.unet.train()

    if global_step >= train_steps:
        break


# ============================================
# 7. Save LoRA weights
# ============================================
os.makedirs(output_dir, exist_ok=True)

pipe.unet.save_pretrained(output_dir)

print("Training finished! LoRA saved to:", output_dir)

wandb.finish()