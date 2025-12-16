# =====================================================
# train_powerpaint_mask.py (Final Version with Input Dir)
# =====================================================

import json
import time
import os
import argparse
from collections import deque 

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

# -------------------------
# Utils
# -------------------------
def denormalize_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor from [-1, 1] range back to [0, 1]."""
    tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0, 1)

# -------------------------
# Dataset
# -------------------------
class SlideCropDataset(Dataset):
    # ⬇️ NEW: 接收 input_dir 參數
    def __init__(self, meta_path, input_dir, image_size=512):
        self.input_dir = input_dir  # 儲存根目錄路徑
        
        with open(meta_path, "r") as f:
            self.samples = json.load(f)["samples"]

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # ⬇️ NEW: 使用 os.path.join 串接 Root Dir 與 JSON 內的相對路徑
        src_path = os.path.join(self.input_dir, s["source"])
        mask_path = os.path.join(self.input_dir, s["mask"])

        # 1. Load Full Images
        try:
            source_full = Image.open(src_path).convert("RGB")
            mask_full = Image.open(mask_path).convert("L")
        except FileNotFoundError as e:
            print(f"[WARN] File not found: {e}")
            # 簡單容錯：若找不到檔案，隨機回傳下一張 (避免訓練中斷)
            return self.__getitem__((idx + 1) % len(self))

        # 2. Crop by BBox
        bbox = s["bbox"] # [left, top, right, bottom]
        source_crop = source_full.crop(bbox)
        mask_crop = mask_full.crop(bbox)

        # 3. Transform
        source_tensor = self.img_tf(source_crop)
        mask_tensor_raw = self.mask_tf(mask_crop)

        # 4. Threshold (Black < 0.01 is editable)
        mask_tensor = (mask_tensor_raw < 0.01).float()

        return {
            "source": source_tensor,
            "mask": mask_tensor,
            "target": source_tensor,
            "prompt": s["prompt"],
        }

# -------------------------
# Main Training Function
# -------------------------
def main(args):
    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.log_dir)

    print(f"[INFO] Loading Model: {args.model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32, 
        safety_checker=None
    ).to(device)

    # Freeze Base Model
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Setup LoRA
    print(f"[INFO] Adding LoRA Adapter (Rank={args.lora_rank})")
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v"],
            lora_dropout=args.lora_dropout,
        )
    )
    pipe.unet.train()

    # ⬇️ NEW: 傳入 input_dir 給 Dataset
    dataset = SlideCropDataset(args.meta_path, args.input_dir, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    scaler = GradScaler()

    loss_ma_buf = deque(maxlen=args.ma_window)
    global_step = 0
    start_time = time.time()
    last_print = time.time()

    print(f"[INFO] Start Training for {args.epochs} Epochs...")

    for epoch in range(args.epochs):
        for batch in loader:
            if time.time() - start_time > args.max_train_time:
                print("[INFO] Reached time limit, saving and exiting.")
                pipe.unet.save_pretrained(args.output_dir)
                writer.close()
                return

            source = batch["source"].to(device)
            mask = batch["mask"].to(device)

            with autocast():
                # VAE Encode
                with torch.no_grad():
                    latents = pipe.vae.encode(source).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                # Add Noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Latent Masking
                mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
                masked_latents = latents * (1 - mask_latent)

                # Text Emb
                text_ids = pipe.tokenizer(
                    batch["prompt"], padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                text_emb = pipe.text_encoder(text_ids)[0]

                # UNet Forward
                unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
                pred = pipe.unet(unet_input, timesteps, encoder_hidden_states=text_emb).sample

                # Loss
                denom = mask_latent.sum().clamp_min(1.0)
                loss = (((pred - noise) ** 2) * mask_latent).sum() / denom

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            loss_val = loss.item()
            loss_ma_buf.append(loss_val)
            loss_ma = sum(loss_ma_buf) / len(loss_ma_buf)

            if global_step % args.tb_log_every == 0:
                writer.add_scalar("train/loss_raw", loss_val, global_step)
                writer.add_scalar("train/loss_ma100", loss_ma, global_step)
                writer.add_scalar("train/mask_coverage", mask_latent.mean().item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % args.print_every == 0:
                now = time.time()
                print(f"[Epoch {epoch}][Step {global_step}] Loss={loss_val:.4f} | MA={loss_ma:.4f} | dt={now - last_print:.1f}s")
                last_print = now
            
            # Visualization
            if global_step % args.visualize_every == 0:
                source_vis = denormalize_to_01(batch["source"][0].cpu())
                target_vis = denormalize_to_01(batch["target"][0].cpu())
                mask_vis = batch["mask"][0].cpu().repeat(3, 1, 1)
                
                grid = make_grid([source_vis, target_vis, mask_vis], nrow=3)
                writer.add_image("train/visualization", grid, global_step)
                print(f"[INFO] Visualized at step {global_step}")

            global_step += 1

    print("[INFO] Training Finished. Saving Model...")
    pipe.unet.save_pretrained(args.output_dir)
    writer.close()
    print(f"[DONE] Saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerPaint LoRA Training Script")
    
    # Paths
    parser.add_argument("--meta_path", type=str, required=True, help="Path to json metadata")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory for images")  # ⬅️ NEW argument
    parser.add_argument("--output_dir", type=str, default="outputs/powerpaint_lora", help="Where to save model")
    parser.add_argument("--log_dir", type=str, default="runs/powerpaint_lora", help="Tensorboard log dir")
    
    # Model & Training
    parser.add_argument("--model_id", type=str, default="Sanster/PowerPaint-V1-stable-diffusion-inpainting")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_time", type=int, default=21600)
    
    # Logging
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--tb_log_every", type=int, default=10)
    parser.add_argument("--visualize_every", type=int, default=500)
    parser.add_argument("--ma_window", type=int, default=100)

    args = parser.parse_args()
    main(args)