import os
import time
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

# Google GenAI SDK (Imagen 3)
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Gemini API key for Imagen 3
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable in your .env")

# Single client for vision/text calls
client = genai.Client(api_key=GEMINI_API_KEY)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMP_DIR = os.path.abspath(os.path.join(ROOT_DIR, "temp"))
os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_TRYON_CATEGORIES = {
    "upper_body",
    "lower_body",
    "dresses",
    "footwear",
    "headwear",
    "accessories",
}

def _explode_labels(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [p.strip().lower() for p in v.split(",") if p.strip()]

import base64

def classify_product(image_path: str, max_retries: int = 3) -> List[str]:
    """
    Upload the image once, then ask the Flash model to extract fashion tags
    directly from the image, skipping the separate vision-describe step.
    """
    try:
        file_obj = client.files.upload(file=image_path)
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                file_obj,
                "Look at this product image and extract a comma-separated list "
                "of fashion-related tags (e.g., 't-shirt', 'blue', 'short-sleeve')."
            ]
        )
        text = getattr(resp, "text", "") or ""
        return _explode_labels(text)
    except Exception as e:
        print(f"[ERROR] classify_product failed: {e}")
        return []

def is_tryon_supported(category: str) -> bool:
    return category in SUPPORTED_TRYON_CATEGORIES

def _garment_desc_from_tags(tags: List[str]) -> str:
    base = []
    for tag in tags:
        t = tag.strip().lower()
        if t in {"t-shirt","shirt","top","tee"} and "t-shirt" not in base:
            base.append("t-shirt")
        elif t in {"hoodie","jacket","coat","blouse","polo"} and t not in base:
            base.append(t)
        elif t in {"pants","jeans","trousers","shorts","leggings"}:
            g = "shorts" if t == "shorts" else "pants"
            if g not in base:
                base.append(g)
        elif t in {"dress","gown"} and "dress" not in base:
            base.append("dress")
        elif t == "skirt" and "skirt" not in base:
            base.append("skirt")
        elif t in {"cap","hat","beanie"} and t not in base:
            base.append(t)
        elif t in {"black","white","red","blue","green"} and t not in base:
            base.append(t)
        elif t in {"round-neck","v-neck","short-sleeve","long-sleeve"} and t not in base:
            base.append(t)
    return ", ".join(base) if base else "fashion garment"

def infer_category(tags: List[str]) -> str:
    s = set(t.strip().lower() for t in tags)
    if {"dress","gown"} & s:
        return "dresses"
    if {"pants","jeans","shorts","skirt"} & s:
        return "lower_body"
    return "upper_body"

def build_prompts(tags: List[str], category: str, view_hint: Optional[str] = None, gender: str = "male") -> Tuple[str, str]:
    view = f", {view_hint}" if view_hint else ""

    if category == "upper_body":
        framing = f"shoulders-to-knees{view}"
    elif category == "lower_body":
        framing = f"hips-to-shoes crop{view}"
    elif category == "dresses":
        framing = f"full body crop{view}"
    elif category == "footwear":
        framing = f"feet-focused crop{view}, ground-in-frame"
    elif category == "headwear":
        framing = f"head-and-shoulders crop{view}"
    else:
        framing = f"knee-up crop{view}"

    style_brief = (
        "no specific gender"
        "high-fashion editorial photoshoot, premium studio lighting, plain grey background, soft key light, crisp details, "
        "rich natural colors, true-to-life color rendering, no monochrome, no black-and-white, "
        "clean seamless backdrop"
        "model ethnicity: indian, asian, or american, european, african, or latin american"
    )
    poses = (
        "dynamic runway-inspired pose, subtle movement, relaxed hands, contrapposto, "
        "fashion lookbook styling like H&M/Vogue"
    )

    if category == "upper_body":
        garment = "a modern top"
    elif category == "lower_body":
        garment = "tailored bottoms"
    elif category == "dresses":
        garment = "a contemporary dress"
    elif category == "footwear":
        garment = "stylish shoes"
    elif category == "headwear":
        garment = "a contemporary hat"
    else:
        garment = "a contemporary outfit"

    pos = (
        f"full-color photo of a {gender} model wearing {garment}, {framing}, "
        f"{style_brief}, {poses}, photorealistic, professional fashion catalog quality"
    )

    neg = (
        "black-and-white, monochrome, sepia, color cast, low contrast, motion blur, watermark, logo, text, "
        "overexposed, underexposed, clutter, busy background, duplicated limbs, distorted anatomy, extra fingers"
    )

    return pos, neg

def _to_imagen_prompt(pos: str, neg: str) -> str:
    return f"{pos}. Avoid: {neg}."

def change_clothes_tryon(
    person_img_path: str,
    garment_img_path: str,
    garment_desc: str,
    category: str,
    steps: int = 40,
    seed: int = -1
) -> Tuple[str, str]:
    """Perform virtual try-on with two-image input via Imagen 3."""
    try:
        with open(person_img_path, "rb") as f:
            person_bytes = f.read()
        with open(garment_img_path, "rb") as f:
            garment_bytes = f.read()

        pos, neg = build_prompts([t for t in garment_desc.split(",")], category)
        prompt = _to_imagen_prompt(pos, neg)

        client_local = genai.Client(api_key=GEMINI_API_KEY)
        cfg = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            output_mime_type="image/png",
            person_generation="allow_adult"
        )
        response = client_local.models.generate_content([
            types.content.ImageData(data=person_bytes),
            types.content.ImageData(data=garment_bytes),
            prompt
        ], config=cfg)

        imgs = getattr(response, "generated_images", None)
        if not imgs:
            raise RuntimeError("No images generated by Imagen 3")
        img_data = imgs[0].image.image_bytes

        out = os.path.join(TEMP_DIR, "final_tryon.png")
        with open(out, "wb") as f:
            f.write(img_data)
        return out, ""
    except Exception as e:
        raise RuntimeError(f"Try-on failed: {e}")

def generate_base_models_imagen3(
    tags: List[str],
    num_images: int = 3,
    forced_category: Optional[str] = None,
    aspect_ratio: str = "1:1"
) -> List[str]:
    category = forced_category or infer_category(tags)
    views = ["front view", "left three-quarter view", "right three-quarter view"]
    client_local = genai.Client(api_key=GEMINI_API_KEY)
    paths = []
    for i in range(num_images):
        pos, neg = build_prompts(tags, category, views[i % 3])
        prompt = _to_imagen_prompt(pos, neg)
        cfg = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            output_mime_type="image/png",
            person_generation="allow_adult"
        )
        resp = client_local.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=cfg
        )
        imgs = getattr(resp, "generated_images", None)
        if not imgs:
            continue
        data = imgs[0].image.image_bytes
        p = os.path.join(TEMP_DIR, f"base_model_{i}.png")
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
    return paths

def generate_studio_models(
    product_tags: List[str],
    num_options: int = 3,
    forced_category: Optional[str] = None,
    aspect_ratio: str = "1:1",
    sample_image_size: str = "1:K"
) -> List[str]:
    """Alias to match your Streamlit import."""
    return generate_base_models_imagen3(
        tags=product_tags,
        num_images=num_options,
        forced_category=forced_category,
        aspect_ratio=aspect_ratio
    )

def run_tryon_with_selected_model(
    selected_model_img_path: str,
    product_image_path: str,
    product_tags: List[str],
    forced_category: Optional[str] = None,
    steps: int = 40
) -> Optional[str]:
    """Wraps change_clothes_tryon and returns the final image path."""
    try:
        cat = forced_category or infer_category(product_tags)
        if not is_tryon_supported(cat):
            return None
        desc = _garment_desc_from_tags(product_tags)
        path, _ = change_clothes_tryon(
            person_img_path=selected_model_img_path,
            garment_img_path=product_image_path,
            garment_desc=desc,
            category=cat,
            steps=steps
        )
        return path
    except Exception as e:
        print(f"[ERROR] run_tryon failed: {e}")
        return None

# Flash-based category classifier
def classify_category_with_flash(image_path: str) -> Optional[str]:
    """
    Use gemini-1.5-flash to classify the uploaded image into one of:
    ['upper_body','lower_body','dresses','footwear','headwear'].
    Returns the slug (e.g., 'upper_body') or None on failure.
    """
    CATEGORIES = ["upper_body", "lower_body", "dresses", "footwear", "headwear"]
    try:
        file_obj = client.files.upload(file=image_path)
        prompt = (
            "Look at the image and pick exactly one category for the MAIN wearable item.\n"
            "Choices (return only the slug): upper_body, lower_body, dresses, footwear, headwear.\n"
            "Rules:\n"
            "- upper_body: shirts, t-shirts, hoodies, jackets, tops.\n"
            "- lower_body: pants, jeans, shorts, skirts, leggings.\n"
            "- dresses: single-piece dress/gown.\n"
            "- footwear: shoes, sneakers, boots, sandals.\n"
            "- headwear: cap, hat, beanie.\n"
            "Return only the slug without extra words."
        )
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[file_obj, prompt]
        )
        text = (getattr(resp, "text", "") or "").strip().lower()
        if text in CATEGORIES:
            return text
        text = text.replace(".", "").replace("\n", "").strip()
        if text in CATEGORIES:
            return text
        alias_map = {
            "upper body": "upper_body",
            "lower body": "lower_body",
            "dress": "dresses",
            "head gear": "headwear",
            "head wear": "headwear",
        }
        if text in alias_map:
            return alias_map[text]
        for c in CATEGORIES:
            if c.replace("_", " ") in text or c in text:
                return c
        return None
    except Exception as e:
        print(f"[ERROR] classify_category_with_flash failed: {e}")
        return None
