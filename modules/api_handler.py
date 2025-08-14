# modules/api_handler.py
import os
import requests
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from typing import List, Tuple, Optional
from PIL import Image

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("Please set the HF_API_TOKEN environment variable in your .env")

HF_MODEL = "google/vit-base-patch16-224"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/octet-stream"}

def _explode_labels(v: str) -> List[str]:
    return [p.strip() for p in v.split(",") if p.strip()]

def classify_product(image_path: str, top_k: int = 10) -> List[str]:
    try:
        with open(image_path, "rb") as f:
            b = f.read()
        r = requests.post(HF_API_URL, headers=HF_HEADERS, data=b, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            return []
        if isinstance(data, dict) and "labels" in data:
            data = data["labels"]
        out: List[str] = []
        if isinstance(data, list):
            for item in data[:top_k]:
                if isinstance(item, dict) and "label" in item:
                    out.extend(_explode_labels(str(item["label"]).lower()))
                elif isinstance(item, str):
                    out.extend(_explode_labels(item.lower()))
        dedup = []
        seen = set()
        for t in out:
            if t not in seen:
                seen.add(t)
                dedup.append(t)
        return dedup
    except Exception:
        return []

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "assets"))
TEMP_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "temp"))
os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_TRYON_CATEGORIES = {"upper_body", "lower_body", "dresses"}

def is_tryon_supported(category: str) -> bool:
    return category in SUPPORTED_TRYON_CATEGORIES

def _garment_desc_from_tags(tags: List[str]) -> str:
    base = []
    for t in tags:
        t = t.strip().lower()
        if t in {"t-shirt", "shirt", "top", "tee"} and "t-shirt" not in base:
            base.append("t-shirt")
        elif t in {"hoodie", "sweatshirt", "sweater", "jacket", "coat", "kurta", "blouse", "polo", "tank", "cardigan"} and t not in base:
            base.append(t)
        elif t in {"pants", "jeans", "trousers", "shorts", "cargo", "chinos", "joggers", "leggings"} and ("pants" if t != "shorts" else "shorts") not in base:
            base.append("shorts" if t == "shorts" else "pants")
        elif t in {"dress", "gown"} and "dress" not in base:
            base.append("dress")
        elif t in {"skirt"} and "skirt" not in base:
            base.append("skirt")
        elif t in {"shoe", "running shoe", "shoes", "sneaker", "sneakers", "boots", "loafers", "heels", "sandal", "clog", "geta", "patten", "sabot", "loafer", "doormat", "welcome mat", "shoe shop", "shoe-shop", "shoe store"} and "sneakers" not in base:
            base.append("sneakers" if "sneaker" in t or t == "sneakers" else "shoes")
        elif t in {"cap", "hat", "beanie", "beret"} and t not in base:
            base.append(t)
        elif t in {"black", "white", "red", "blue", "green", "yellow", "beige", "brown", "grey", "gray"} and t not in base:
            base.append(t)
        elif t in {"round-neck", "v-neck", "collared"} and t not in base:
            base.append(t)
        elif t in {"short-sleeve", "long-sleeve"} and t not in base:
            base.append(t)
        elif t in {"slim-fit", "oversized", "regular-fit"} and t not in base:
            base.append(t)
    return ", ".join(base) if base else "fashion garment"

def infer_category(tags: List[str]) -> str:
    s = {t.strip().lower() for t in tags}
    if {"dress", "gown"} & s:
        return "dresses"
    if {"cap", "hat", "beanie", "beret"} & s:
        return "headwear"
    if {"shoe", "shoes", "sneaker", "sneakers", "boots", "loafers", "heels", "sandal", "clog", "geta", "patten", "sabot", "loafer", "doormat", "welcome mat", "shoe shop", "shoe-shop", "shoe store"} & s:
        return "footwear"
    if {"pants", "trousers", "jeans", "jean", "blue jean", "denim", "shorts", "skirt", "cargo", "chinos", "joggers", "leggings"} & s:
        return "lower_body"
    if {"t-shirt", "tee", "shirt", "top", "hoodie", "sweatshirt", "sweater", "jacket", "coat", "kurta", "blouse", "polo", "tank", "cardigan"} & s:
        return "upper_body"
    return "upper_body"

SD_SPACE = "stabilityai/stable-diffusion"
SD_API = "/infer"

def build_prompts(tags: List[str], category: str, view_hint: Optional[str] = None, gender: str = "male") -> Tuple[str, str]:
    desc = _garment_desc_from_tags(tags)
    v = f", {view_hint}" if view_hint else ""
    if category == "upper_body":
        framing = f"tight shoulders-to-waist crop{v}, face out of frame, focus on chest, sleeves and torso"
        outfit = "plain close-fit neutral top"
        pose = "arms slightly away from torso"
        negative = "face, eyes, head, text, watermark, logo, multiple people, clutter, blur, low-res, artifacts, bad anatomy, extra limbs, overexposed, underexposed"
    elif category == "lower_body":
        framing = f"full view from hips to shoes{v}, torso cropped above hips, focus on pants and legs"
        outfit = "plain neutral fitted pants"
        pose = "standing straight, legs visible, feet shoulder-width"
        negative = "upper body, bare chest, shirt, t-shirt, hoodie, jacket, torso, face, head, text, watermark, logo, multiple people, clutter, blur, low-res, artifacts, bad anatomy, extra limbs, overexposed, underexposed"
    elif category == "dresses":
        framing = f"knee-up crop{v}, full dress silhouette in frame"
        outfit = "plain neutral dress"
        pose = "hands relaxed by sides"
        negative = "text, watermark, logo, multiple people, clutter, blur, low-res, artifacts, bad anatomy, extra limbs, overexposed, underexposed"
    elif category == "footwear":
        framing = f"close-up feet and lower legs{v}, shoes centered, entire shoe visible"
        outfit = "neutral ankle-length pants exposing shoes"
        pose = "standing, feet flat on ground"
        negative = "text, watermark, logo, multiple people, clutter, blur, low-res, artifacts, overexposed, underexposed"
    elif category == "headwear":
        framing = f"tight head-and-shoulders crop{v}, headwear centered"
        outfit = "plain neutral top with simple neckline"
        pose = "neutral expression"
        negative = "text, watermark, logo, multiple people, clutter, blur, low-res, artifacts, overexposed, underexposed"
    else:
        framing = f"shoulders-to-waist crop{v}, face out of frame"
        outfit = "plain neutral garment"
        pose = "neutral"
        negative = "text, watermark, logo, multiple people, clutter, blur, low-res, artifacts, overexposed, underexposed"
    positive = f"photo of a {gender} fashion model, {framing}, {outfit}, studio lighting, soft shadows, high detail, 85mm look, seamless backdrop, {pose}, sharp focus, photorealistic, emphasizing {desc}"
    return positive, negative

def _tp(name: str) -> str:
    return os.path.abspath(os.path.join(TEMP_DIR, name))

def generate_base_models_sd(tags: List[str], num_images: int = 3, guidance_scale: float = 9.0, forced_category: Optional[str] = None) -> List[str]:
    cat = forced_category if forced_category else infer_category(tags)
    views = ["front view", "left three-quarter view", "right three-quarter view"]
    client = Client(SD_SPACE, hf_token=HF_API_TOKEN)
    out: List[str] = []
    for i in range(num_images):
        p, n = build_prompts(tags, cat, views[i % len(views)])
        try:
            res = client.predict(prompt=p, negative=n, scale=float(guidance_scale), api_name=SD_API)
            if not isinstance(res, list) or not res:
                continue
            src = res[0].get("image") if isinstance(res[0], dict) else None
            if not src:
                continue
            dst = _tp(f"base_model_sd_{i}.png")
            try:
                with open(src, "rb") as s, open(dst, "wb") as d:
                    d.write(s.read())
            except Exception:
                dst = os.path.abspath(src)
            out.append(dst)
        except Exception:
            continue
    return out

TRYON_SPACE = "jallenjia/Change-Clothes-AI"
TRYON_API = "/tryon"

def preprocess_garment(src: str) -> str:
    im = Image.open(src).convert("RGB")
    im.thumbnail((768, 768))
    path = _tp("garment_clean.png")
    im.save(path, "PNG")
    return path

def _resolve_person(p: str) -> str:
    if p and os.path.exists(p):
        return os.path.abspath(p)
    b = os.path.basename(p) if p else ""
    c = _tp(b)
    if os.path.exists(c):
        return c
    for i in range(3):
        f = _tp(f"base_model_sd_{i}.png")
        if os.path.exists(f):
            return f
    raise FileNotFoundError("base_model_sd_[0-2].png not found in temp")

def change_clothes_tryon(
    person_img_path: str,
    garment_img_path: str,
    garment_desc: str,
    category: str,
    auto_mask: bool = True,
    auto_crop: bool = True,
    steps: int = 40,
    seed: int = -1
) -> Tuple[str, str]:
    client = Client(TRYON_SPACE, hf_token=HF_API_TOKEN)
    person_img_path = _resolve_person(person_img_path)
    if not os.path.exists(garment_img_path):
        raise FileNotFoundError(garment_img_path)
    payload = {"background": handle_file(person_img_path), "layers": [], "composite": None, "id": None}
    res = client.predict(
        dict=payload,
        garm_img=handle_file(garment_img_path),
        garment_des=garment_desc,
        is_checked=auto_mask,
        is_checked_crop=auto_crop,
        denoise_steps=float(steps),
        seed=float(seed),
        category=category,
        api_name=TRYON_API
    )
    if isinstance(res, dict) and "error" in res:
        raise RuntimeError(res["error"])
    if not isinstance(res, (list, tuple)) or not res:
        raise RuntimeError("empty response")
    a = str(res[0])
    b = str(res[1]) if len(res) > 1 else ""
    return a, b

def generate_studio_models(product_tags: List[str], num_options: int = 3, forced_category: Optional[str] = None) -> List[str]:
    return generate_base_models_sd(product_tags, num_images=num_options, guidance_scale=9.0, forced_category=forced_category)

def run_tryon_with_selected_model(
    selected_model_img_path: str,
    product_image_path: str,
    product_tags: List[str],
    forced_category: Optional[str] = None,
    steps: int = 40
) -> Optional[str]:
    try:
        desc = _garment_desc_from_tags(product_tags)
        cat = forced_category if forced_category else infer_category(product_tags)
        if not is_tryon_supported(cat):
            return None
        g = preprocess_garment(product_image_path)
        final_path, _ = change_clothes_tryon(
            person_img_path=selected_model_img_path,
            garment_img_path=g,
            garment_desc=desc,
            category=cat,
            auto_mask=True,
            auto_crop=True,
            steps=steps,
            seed=-1
        )
        out = _tp("final_tryon.png")
        try:
            with open(final_path, "rb") as s, open(out, "wb") as d:
                d.write(s.read())
        except Exception:
            out = os.path.abspath(final_path)
        return out
    except Exception:
        return None
