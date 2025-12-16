# with_yolo_inference.py
"""
WITH-YOLO INFERENCE (YOLO bbox + GT mask)

Pipeline:
YOLO -> bounding box
GT mask (from meta.json) -> editable pixels (BLACK = editable)
PowerPaint (+ optional LoRA) -> edit by prompt
"""

import os
import json
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# -------------------------------------------------
# HF compatibility patch (cached_download removal)
# -------------------------------------------------
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download

    def cached_download(*args, **kwargs):
        return hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = cached_download


from diffusers import StableDiffusionInpaintPipeline
from ultralytics import YOLO

# Optional PEFT (LoRA)
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

import lpips
import clip


# -------------------------------------------------
# Utils
# -------------------------------------------------
def build_experiment_name(args) -> str:
    lora_part = "w_lora" if args.with_lora else "wo_lora"
    data_part = "all" if args.num_input is None else str(args.num_input)
    save_part = "save_all" if args.save_all else f"vis_{args.num_vis}"
    return f"{lora_part}_{data_part}_{save_part}_seed{args.seed}"


def extract_sample_id(source_filename: str) -> str:
    # e.g. 001_9_source_001.png -> 001_9
    return os.path.basename(source_filename).split("_source_")[0]


def save_sample_bundle(
    root: str,
    sample_id: str,
    prompt: str,
    source_slide: Image.Image,
    target_slide: Image.Image,
    output_slide: Image.Image,
    source_crop: Image.Image,
    mask_crop: Image.Image,
):
    sample_dir = os.path.join(root, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    source_slide.save(os.path.join(sample_dir, "source_slide.png"))
    target_slide.save(os.path.join(sample_dir, "target_slide.png"))
    output_slide.save(os.path.join(sample_dir, "output_slide.png"))
    source_crop.save(os.path.join(sample_dir, "source_crop.png"))
    mask_crop.save(os.path.join(sample_dir, "mask_crop.png"))

    with open(os.path.join(sample_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)


def paste_crop(slide: Image.Image, crop: Image.Image, bbox):
    slide = slide.copy()
    slide.paste(crop, bbox[:2])
    return slide


def compute_mean_std(values):
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _to_lpips_range(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def _clip_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum(dim=-1).item())


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # -----------------------------
    # Output directory
    # -----------------------------
    exp_name = build_experiment_name(args)
    out_root = os.path.join(args.out_dir, exp_name)
    os.makedirs(out_root, exist_ok=True)

    print(f"[INFO] Output dir: {out_root}")

    # -----------------------------
    # Load meta
    # -----------------------------
    with open(args.meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    if args.num_input is not None and args.num_input < len(samples):
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.num_input)

    print(f"[INFO] Running inference on {len(samples)} samples")

    # -----------------------------
    # Decide save set
    # -----------------------------
    all_ids = [extract_sample_id(s["source"]) for s in samples]
    if args.save_all:
        save_ids = set(all_ids)
    else:
        rng = random.Random(args.seed)
        save_ids = set(rng.sample(all_ids, k=min(args.num_vis, len(all_ids))))

    # -----------------------------
    # Load YOLO
    # -----------------------------
    print(f"[INFO] Loading YOLO from {args.yolo_pt}")
    yolo = YOLO(args.yolo_pt)

    # -----------------------------
    # Load PowerPaint
    # -----------------------------
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    if args.with_lora:
        if not args.model_path:
            raise ValueError("--with_lora requires --model_path")
        if PeftModel is None:
            raise ImportError("peft not installed")

        print(f"[INFO] Loading LoRA from {args.model_path}")
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            args.model_path,
            is_trainable=False,
        )

    for m in [pipe.unet, pipe.vae, getattr(pipe, "text_encoder", None)]:
        if m is not None and hasattr(m, "eval"):
            m.eval()

    # -----------------------------
    # Metrics models
    # -----------------------------
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    tf_crop_rgb = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
    ])
    tf_mask = transforms.Compose([
        transforms.Resize(
            (args.crop_size, args.crop_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.ToTensor(),
    ])
    tf_slide = transforms.ToTensor()

    results = []
    masked_lpips_all, clip_score_all, bg_lpips_all, clip_i_all = [], [], [], []

    # -----------------------------
    # Inference loop
    # -----------------------------
    for s in tqdm(samples, desc="With-YOLO (GT mask) Inference"):
        source_path = os.path.join(args.img_dir, s["source"])
        target_path = os.path.join(args.img_dir, s["target"])
        mask_path   = os.path.join(args.img_dir, s["mask"])

        if not (os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(mask_path)):
            continue

        sample_id = extract_sample_id(s["source"])
        prompt = s["prompt"]

        source_slide = Image.open(source_path).convert("RGB")
        target_slide = Image.open(target_path).convert("RGB")
        mask_slide   = Image.open(mask_path).convert("L")  # GT mask

        # -------------------------
        # YOLO detect -> bbox
        # -------------------------
        yolo_res = yolo(source_slide, conf=args.yolo_conf, verbose=False)[0]
        if yolo_res.boxes is None or len(yolo_res.boxes) == 0:
            continue

        boxes  = yolo_res.boxes.xyxy.cpu().numpy()
        scores = yolo_res.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(scores))
        bbox   = boxes[best_i].astype(int).tolist()

        # -------------------------
        # Crop image + GT mask
        # -------------------------
        source_crop = source_slide.crop(bbox)
        raw_mask_crop = mask_slide.crop(bbox)

        # GT rule: BLACK (0) = editable
        raw_np = np.array(raw_mask_crop, dtype=np.uint8)
        editable = (raw_np == 0)

        diffusers_mask_np = np.zeros_like(raw_np, dtype=np.uint8)
        diffusers_mask_np[editable] = 255

        mask_crop = Image.fromarray(diffusers_mask_np, mode="L")

        # -------------------------
        # PowerPaint inference
        # -------------------------
        with torch.no_grad():
            out_crop = pipe(
                prompt=prompt,
                image=source_crop,
                mask_image=mask_crop,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ).images[0]

        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if out_crop.size != (bw, bh):
            out_crop = out_crop.resize((bw, bh), Image.BICUBIC)

        output_slide = paste_crop(source_slide, out_crop, bbox)

        # -------------------------
        # Metrics
        # -------------------------
        out_t = tf_crop_rgb(out_crop).unsqueeze(0).to(device)
        in_t  = tf_crop_rgb(source_crop).unsqueeze(0).to(device)
        m_t   = tf_mask(mask_crop).unsqueeze(0).to(device)
        m_t3  = m_t.repeat(1, 3, 1, 1)

        masked_lpips = lpips_fn(
            _to_lpips_range(out_t) * m_t3,
            _to_lpips_range(in_t)  * m_t3,
        ).mean().item()

        clip_img = clip_preprocess(out_crop).unsqueeze(0).to(device)
        clip_txt = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            clip_score = _clip_cosine(
                clip_model.encode_image(clip_img),
                clip_model.encode_text(clip_txt),
            )

        out_slide_t = tf_slide(output_slide).unsqueeze(0).to(device)
        in_slide_t  = tf_slide(source_slide).unsqueeze(0).to(device)
        bg_mask     = 1.0 - tf_slide(mask_slide).unsqueeze(0).to(device)
        bg_mask3    = bg_mask.repeat(1, 3, 1, 1)

        bg_lpips = lpips_fn(
            _to_lpips_range(out_slide_t) * bg_mask3,
            _to_lpips_range(in_slide_t)  * bg_mask3,
        ).mean().item()

        clip_out = clip_preprocess(output_slide).unsqueeze(0).to(device)
        clip_gt  = clip_preprocess(target_slide).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_i = _clip_cosine(
                clip_model.encode_image(clip_out),
                clip_model.encode_image(clip_gt),
            )

        results.append({
            "sample_id": sample_id,
            "bbox": bbox,
            "yolo_conf": float(scores[best_i]),
            "masked_lpips": masked_lpips,
            "clip_score": clip_score,
            "bg_lpips": bg_lpips,
            "clip_i": clip_i,
        })

        masked_lpips_all.append(masked_lpips)
        clip_score_all.append(clip_score)
        bg_lpips_all.append(bg_lpips)
        clip_i_all.append(clip_i)

        # -------------------------
        # Save
        # -------------------------
        if sample_id in save_ids:
            save_sample_bundle(
                out_root,
                sample_id,
                prompt,
                source_slide,
                target_slide,
                output_slide,
                source_crop,
                mask_crop,
            )

    # -----------------------------
    # Save summary
    # -----------------------------
    with open(os.path.join(out_root, "results_with_yolo.json"), "w") as f:
        json.dump(results, f, indent=2)

    m_lpips, s_lpips = compute_mean_std(masked_lpips_all)
    m_clip,  s_clip  = compute_mean_std(clip_score_all)
    m_bg,    s_bg    = compute_mean_std(bg_lpips_all)
    m_ci,    s_ci    = compute_mean_std(clip_i_all)

    summary = {
        "split": args.split,
        "n": len(results),
        "with_lora": args.with_lora,
        "masked_lpips": {"mean": m_lpips, "std": s_lpips},
        "clip_score":   {"mean": m_clip,  "std": s_clip},
        "bg_lpips":     {"mean": m_bg,    "std": s_bg},
        "clip_i":       {"mean": m_ci,    "std": s_ci},
    }

    with open(os.path.join(out_root, "summary_with_yolo.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Results saved to {out_root}")


# -------------------------------------------------
# Args
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./with_yolo_results")
    parser.add_argument("--split", type=str, required=True)

    parser.add_argument("--yolo_pt", type=str, default="./best/best.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.25)

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--with_lora", action="store_true")

    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--num_input", type=int, default=None)
    parser.add_argument("--num_vis", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_all", action="store_true")

    args = parser.parse_args()
    main(args)
