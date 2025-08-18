import os
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

from google import genai
from google.genai import types

import random

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set the GEMINI_API_KEY environment variable in your .env")

client = genai.Client(api_key=GEMINI_API_KEY)

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

ETHNICITY_TONE_PROMPTS = [
    "South Asian model with warm brown skin tone",
    "East Asian model with light golden skin tone",
    "Northern European model with cool porcelain skin tone",
    "Eastern European model with fair neutral-beige skin tone",
    "Western European model with light rosy skin tone",
    "Southern European model with warm olive-beige skin tone",
    "Mediterranean European model with sun-kissed bronze skin tone",
    "African American model with rich espresso skin tone",
    "Latino/Hispanic American model with golden tan skin tone",
    "Pakistani model with warm medium-brown skin tone",
]

def _random_demographic(gender: str) -> Tuple[str, str]:
    g = (gender or "").strip().lower()
    if g not in {"male", "female"}:
        g = "male"
    desc = random.choice(ETHNICITY_TONE_PROMPTS)
    return g, desc

def _explode_labels(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [p.strip().lower() for p in v.split(",") if p.strip()]

def classify_product(image_path: str, max_retries: int = 3) -> List[str]:
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
        framing = f"shoulders-to-waist{view}"
        garment = "clean plain crewneck t-shirt (no logos), neutral mid-grey"
    elif category == "lower_body":
        framing = f"hips-to-shoes{view}"
        garment = "plain slim chinos (no logos), neutral charcoal"
    elif category == "dresses":
        framing = f"knee-up{view}"
        garment = "minimal solid-color dress, neutral grey"
    elif category == "footwear":
        framing = f"feet-focused close-up{view}, ground in frame"
        garment = "ankle socks only (no shoes)"
    elif category == "headwear":
        framing = f"head-and-shoulders{view}"
        garment = "no hat"
    else:
        framing = f"knee-up{view}"
        garment = "simple neutral outfit"

    style_brief = (
        "studio fashion look, premium soft key light, plain medium-grey seamless backdrop, "
        "true-to-life colors, crisp details, no monochrome"
    )
    poses = "relaxed lookbook pose, natural hands, fashion catalog quality"

    g_label, demo_desc = _random_demographic(gender)

    pos = (
        f"full-color photo of a {g_label} {demo_desc}, {framing}, wearing {garment}, "
        f"{style_brief}, {poses}, photorealistic"
    )
    neg = (
        "logos, text, brand marks, stripes, patterns, busy background, color cast, harsh HDR, "
        "underexposed, overexposed, motion blur, duplicated limbs, distorted anatomy"
    )
    return pos, neg

def _to_imagen_prompt(pos: str, neg: str) -> str:
    return f"{pos}. Avoid: {neg}."

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
    return generate_base_models_imagen3(
        tags=product_tags,
        num_images=num_options,
        forced_category=forced_category,
        aspect_ratio=aspect_ratio
    )

def _build_tryon_instruction_prompt(garment_desc: str, category: str) -> str:
    return (
        "Virtual try-on compositing.\n"
        "Image B = FIRST image (BASE/CANVAS): the person/model photo.\n"
        "Image A = SECOND image (TOP/OVERLAY): the garment/product photo.\n"
        "Task: Render Image B wearing the garment from Image A.\n"
        f"Garment details: {garment_desc}. Category: {category}.\n"
        "CRITICAL DESIGN FIDELITY from Image A: exact colors and ratios; stripe/block layout and stripe thickness; "
        "graphics/prints/numbering text content and typography; logo placement and scale; collar type/color; "
        "placket/button count and spacing; cuff/hem colors; seam/panel layout. Do not simplify or invent new graphics. "
        "Preserve print placements (left chest, center front, etc.) and keep edges crisp.\n"
        "HARD CONSTRAINTS from Image B: keep background, composition, framing/crop, camera angle, lens look, "
        "pose, body shape, skin tone, hair, hands, and environment unchanged. "
        "Do not change lighting or color grading; preserve Image B’s tonal range, white balance, shadow direction/softness, and contrast. "
        "No background replacement or cleanup to white.\n"
        "Fit realism: align prints with body perspective; maintain continuity across seams; realistic drape and stretch at shoulders/chest/sleeves; "
        "natural wrinkles; no blur or washout of prints.\n"
        "If any conflict arises, prioritize preserving the garment design from Image A while keeping pose/lighting/background from Image B.\n"
        "If footwear, keep leg/foot pose and ground contact identical to Image B; match cast shadows; do not alter the floor/background.\n"
        "Output one photorealistic image only. No added text, watermarks, borders, extra limbs, or artifacts."
    )

def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def change_clothes_tryon_virtual_flash_preview(
    person_img_path: str,
    garment_img_path: str,
    garment_desc: str,
    category: str,
    out_name: str = "final_tryon.png"
) -> Tuple[str, str]:
    try:
        def sniff_mime(p: str) -> str:
            ext = os.path.splitext(p)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                return "image/jpeg"
            return "image/png"

        person_bytes = _read_image_bytes(person_img_path)
        garment_bytes = _read_image_bytes(garment_img_path)
        person_mime = sniff_mime(person_img_path)
        garment_mime = sniff_mime(garment_img_path)

        instruction = (
            "Virtual try-on compositing.\n"
            "Image B = FIRST image (BASE/CANVAS): the person/model photo.\n"
            "Image A = SECOND image (TOP/OVERLAY): the garment/product photo.\n"
            "Task: Render Image B wearing the garment from Image A.\n"
            f"Garment details: {garment_desc}. Category: {category}.\n"
            "CRITICAL DESIGN FIDELITY (must be preserved from Image A): "
            "exact colors and color ratios, stripe/block layout and stripe thickness, "
            "graphics/prints/numbering text content and typography, logo placement and scale, "
            "collar type and color, placket/button count and spacing, cuff/hem colors, seams and panels. "
            "Do not simplify patterns, do not shift print positions, and do not invent new graphics.\n"
            "HARD CONSTRAINTS from Image B: background, composition, framing/crop, camera angle, "
            "lens look, pose, body shape, skin tone, hair, hands, and environment. "
            "Lighting and color grading must match Image B exactly: same tonal range, white balance, "
            "shadow direction/softness, and contrast. No background replacement.\n"
            "Fit: align stripes/prints with the body perspective; maintain continuity across seams; "
            "realistic drape and stretch/compression at shoulders, chest, and sleeves; natural wrinkles; "
            "do not blur or wash out prints; keep edges crisp.\n"
            "If any conflict arises, prioritize preserving the garment’s design (from Image A) while keeping "
            "pose/lighting/background from Image B.\n"
            "If footwear, keep leg/foot pose and ground contact identical to Image B; match cast shadows; "
            "do not alter the floor/background.\n"
            "Output one photorealistic image only. no borders, extra limbs, or artifacts."
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=instruction),
                    types.Part.from_bytes(mime_type=person_mime, data=person_bytes),
                    types.Part.from_bytes(mime_type=garment_mime, data=garment_bytes),
                ],
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"]
        )

        model_name = "gemini-2.0-flash-exp"

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )

        image_bytes = None
        if getattr(response, "candidates", None):
            for cand in response.candidates:
                for p in getattr(cand.content, "parts", []) or []:
                    inline = getattr(p, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        image_bytes = inline.data
                        break
                if image_bytes:
                    break

        if not image_bytes:
            debug_text = getattr(response, "text", "") or "No image returned by the model."
            raise RuntimeError(debug_text)

        out_path = os.path.join(TEMP_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return out_path, ""

    except Exception as e:
        return "", f"Virtual try-on failed: {e}"

def change_clothes_tryon(
    person_img_path: str,
    garment_img_path: str,
    garment_desc: str,
    category: str,
    steps: int = 40,
    seed: int = -1
) -> Tuple[str, str]:
    return "", "Legacy Imagen 3 try-on is disabled."

def run_tryon_with_selected_model(
    selected_model_img_path: str,
    product_image_path: str,
    product_tags: List[str],
    forced_category: Optional[str] = None,
    steps: int = 40
) -> Optional[str]:
    try:
        cat = forced_category or infer_category(product_tags)
        if not is_tryon_supported(cat):
            return None
        desc = _garment_desc_from_tags(product_tags)
        out_path, err = change_clothes_tryon_virtual_flash_preview(
            person_img_path=selected_model_img_path,
            garment_img_path=product_image_path,
            garment_desc=desc,
            category=cat,
            out_name="final_tryon.png",
        )
        if err:
            print(f"[ERROR] run_tryon (Flash Preview) failed: {err}")
            return None
        return out_path
    except Exception as e:
        print(f"[ERROR] run_tryon failed: {e}")
        return None

def classify_category_with_flash(image_path: str) -> Optional[str]:
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
        text = text.replace(".", "").replace("\\n", "").strip()
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
