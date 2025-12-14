# no_yolo_inference.py

"""
============================================================
NO-YOLO INFERENCE (Oracle Bounding Box Evaluation)
============================================================

這版假設資料結構為：

- meta：gt/<split>/meta.json
- 影像：dataset/<split>/ 底下同時包含
    - *_source_*.png
    - *_target_*.png
    - *_mask_*.png

meta.json 格式：

{
  "samples": [
    {
      "source": "001_9_source_001.png",
      "target": "001_9_target_001.png",
      "mask":   "001_9_mask_001.png",
      "bbox":   [x1, y1, x2, y2],
      "prompt": "editing instruction text"
    },
    ...
  ]
}
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

from diffusers import StableDiffusionInpaintPipeline

# peft 是「選用」的：只有在需要載入 LoRA (--model_path) 時才真的用到
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

import lpips
import clip


# -------------------------------------------------
# Utils
# -------------------------------------------------
def paste_crop(slide: Image.Image, crop: Image.Image, bbox):
    """Paste crop back to slide at (x1, y1). Assumes crop size matches bbox size."""
    slide = slide.copy()
    slide.paste(crop, bbox[:2])
    return slide


def compute_mean_std(values):
    arr = np.array(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _to_lpips_range(x01: torch.Tensor) -> torch.Tensor:
    """LPIPS expects inputs in [-1, 1]. Input assumed in [0, 1]."""
    return x01 * 2.0 - 1.0


def _clip_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two feature tensors."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return float((a * b).sum(dim=-1).item())


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    os.makedirs(args.out_dir, exist_ok=True)
    vis_root = os.path.join(args.out_dir, "vis")
    os.makedirs(vis_root, exist_ok=True)

    # -----------------------------
    # Load meta
    # -----------------------------
    with open(args.meta_path, "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    # -----------------------------
    # RNG for visualization sampling
    # -----------------------------
    rng = random.Random(args.seed)
    vis_indices = set(rng.sample(range(len(samples)), k=min(args.num_vis, len(samples))))

    # -----------------------------
    # Load PowerPaint
    # -----------------------------
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "Sanster/PowerPaint-V1-stable-diffusion-inpainting",
        torch_dtype=dtype,
    ).to(device)

    # 只有在真的有指定 --model_path 時才會用到 PeftModel
    if args.model_path is not None and args.model_path != "":
        if PeftModel is None:
            raise ImportError(
                "你有指定 --model_path 但環境裡沒有安裝 peft。\n"
                "請先安裝：pip install peft"
            )
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.model_path)

    pipe.set_progress_bar_config(disable=True)
    pipe.eval()

    # -----------------------------
    # Metrics models
    # -----------------------------
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

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

    # -----------------------------
    # Storage
    # -----------------------------
    results = []
    masked_lpips_all, clip_score_all, bg_lpips_all, clip_i_all = [], [], [], []

    # -----------------------------
    # Inference loop
    # -----------------------------
    for idx, s in enumerate(tqdm(samples, desc="No-YOLO Inference")):
        # 所有檔案都在同一個 img_dir 底下
        source_path = os.path.join(args.img_dir, s["source"])
        target_path = os.path.join(args.img_dir, s["target"])
        mask_path   = os.path.join(args.img_dir, s["mask"])

        if not (os.path.exists(source_path) and os.path.exists(target_path) and os.path.exists(mask_path)):
            print(f"⚠ 缺少檔案，跳過 idx={idx}:")
            print(f"   source: {source_path}")
            print(f"   target: {target_path}")
            print(f"   mask  : {mask_path}")
            continue

        source_slide = Image.open(source_path).convert("RGB")
        target_slide = Image.open(target_path).convert("RGB")
        mask_slide   = Image.open(mask_path).convert("L")

        bbox = s["bbox"]
        prompt = s["prompt"]

        # Crop region using oracle bbox
        source_crop = source_slide.crop(bbox)
        mask_crop = mask_slide.crop(bbox)

        # Ensure binary mask (0 or 255)
        mask_crop = mask_crop.point(lambda p: 255 if p > 127 else 0)

        # -------------------------
        # PowerPaint inference (crop-level)
        # -------------------------
        with torch.no_grad():
            out_crop = pipe(
                prompt=prompt,
                image=source_crop,
                mask_image=mask_crop,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
            ).images[0]

        # Resize edited crop back to bbox size (if model outputs different size)
        bw, bh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        if out_crop.size != (bw, bh):
            out_crop = out_crop.resize((bw, bh), resample=Image.BICUBIC)

        # Paste back to slide
        output_slide = paste_crop(source_slide, out_crop, bbox)

        # -------------------------
        # Metric 1: Masked LPIPS (crop)
        # -------------------------
        out_t = tf_crop_rgb(out_crop).unsqueeze(0).to(device)
        in_t = tf_crop_rgb(source_crop).unsqueeze(0).to(device)

        m_t = tf_mask(mask_crop).unsqueeze(0).to(device)  # (1,1,H,W)
        m_t3 = m_t.repeat(1, 3, 1, 1)

        masked_lpips = lpips_fn(
            _to_lpips_range(out_t) * m_t3,
            _to_lpips_range(in_t) * m_t3,
        ).mean().item()

        # -------------------------
        # Metric 2: CLIP score (image-text, crop)
        # -------------------------
        clip_img = clip_preprocess(out_crop).unsqueeze(0).to(device)
        clip_txt = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(clip_img)
            txt_feat = clip_model.encode_text(clip_txt)
            clip_score = _clip_cosine(img_feat, txt_feat)

        # -------------------------
        # Metric 3: Background LPIPS (slide)
        # -------------------------
        out_slide_t = tf_slide(output_slide).unsqueeze(0).to(device)
        in_slide_t = tf_slide(source_slide).unsqueeze(0).to(device)

        bg_mask = 1.0 - tf_slide(mask_slide).unsqueeze(0).to(device)  # (1,1,H,W)
        bg_mask3 = bg_mask.repeat(1, 3, 1, 1)

        bg_lpips = lpips_fn(
            _to_lpips_range(out_slide_t) * bg_mask3,
            _to_lpips_range(in_slide_t) * bg_mask3,
        ).mean().item()

        # -------------------------
        # Metric 4: CLIP-I (image-image, slide)
        # -------------------------
        clip_out = clip_preprocess(output_slide).unsqueeze(0).to(device)
        clip_gt = clip_preprocess(target_slide).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_out = clip_model.encode_image(clip_out)
            feat_gt = clip_model.encode_image(clip_gt)
            clip_i = _clip_cosine(feat_out, feat_gt)

        # -------------------------
        # Record
        # -------------------------
        rec = {
            "id": idx,
            "file": {
                "source": s["source"],
                "target": s["target"],
                "mask": s["mask"],
            },
            "bbox": bbox,
            "prompt": prompt,
            "masked_lpips": masked_lpips,
            "clip_score": clip_score,
            "bg_lpips": bg_lpips,
            "clip_i": clip_i,
        }
        results.append(rec)

        masked_lpips_all.append(masked_lpips)
        clip_score_all.append(clip_score)
        bg_lpips_all.append(bg_lpips)
        clip_i_all.append(clip_i)

        # -------------------------
        # Visualization
        # -------------------------
        if idx in vis_indices:
            vis_dir = os.path.join(vis_root, f"{idx:04d}")
            os.makedirs(vis_dir, exist_ok=True)

            source_slide.save(os.path.join(vis_dir, "source_slide.png"))
            target_slide.save(os.path.join(vis_dir, "target_slide.png"))
            output_slide.save(os.path.join(vis_dir, "output_slide.png"))

    # -----------------------------
    # Save per-sample results
    # -----------------------------
    results_path = os.path.join(args.out_dir, "results_no_yolo.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # -----------------------------
    # Summary (mean/std)
    # -----------------------------
    m_lpips, s_lpips = compute_mean_std(masked_lpips_all)
    m_clip, s_clip = compute_mean_std(clip_score_all)
    m_bg, s_bg = compute_mean_std(bg_lpips_all)
    m_ci, s_ci = compute_mean_std(clip_i_all)

    summary = {
        "setting": "no_yolo (oracle bbox)",
        "n": len(results),
        "masked_lpips": {"mean": m_lpips, "std": s_lpips},
        "clip_score": {"mean": m_clip, "std": s_clip},
        "bg_lpips": {"mean": m_bg, "std": s_bg},
        "clip_i": {"mean": m_ci, "std": s_ci},
        "args": {
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "crop_size": args.crop_size,
            "model_path": args.model_path,
            "seed": args.seed,
            "num_vis": args.num_vis,
            "img_dir": args.img_dir,
            "meta_path": args.meta_path,
        },
    }

    summary_path = os.path.join(args.out_dir, "summary_no_yolo.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------
    # Console summary
    # -----------------------------
    print("\n[No-YOLO Oracle Evaluation]")
    print(f"Samples: {summary['n']}")
    print(f"Masked LPIPS (edit strength): {summary['masked_lpips']['mean']:.4f} ± {summary['masked_lpips']['std']:.4f}")
    print(f"CLIP score (instruction):    {summary['clip_score']['mean']:.4f} ± {summary['clip_score']['std']:.4f}")
    print(f"BG LPIPS (preservation):     {summary['bg_lpips']['mean']:.4f} ± {summary['bg_lpips']['std']:.4f}")
    print(f"CLIP-I (GT similarity):      {summary['clip_i']['mean']:.4f} ± {summary['clip_i']['std']:.4f}")


# -------------------------------------------------
# Args
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="No-YOLO (oracle bbox) inference for PowerPaint"
    )

    # Dataset inputs
    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to meta JSON file (contains filenames, bbox, prompt)",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing slide-level images (source/target/mask all here)",
    )

    # Optional model / output
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path. If omitted, use pretrained PowerPaint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./no_yolo_results",
        help="Output directory for results and visualizations",
    )

    # Inference hyperparameters
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion inference steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )

    # Misc
    parser.add_argument(
        "--crop_size",
        type=int,
        default=512,
        help="Resize size for crop-level metric computation",
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=5,
        help="Number of random samples to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for visualization sampling",
    )

    args = parser.parse_args()
    main(args)
