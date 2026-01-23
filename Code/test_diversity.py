import torch
import cv2
import numpy as np
import os
import random
import re
from PIL import Image
import albumentations as A
from diffusers import StableDiffusionGLIGENPipeline
from tqdm import tqdm

# --- Directory Configuration ---
CATALOG_DIR = "catalog"               # Source of clean images
OUTPUT_DIR = "test_dataset_unseen"    # Where synthetic CCTV images will be stored
CASE_TEMPLATE_PATH = "case_template.jpg"
LOCAL_BG_DIR = "my_backgrounds"       # Backgrounds to make the scene realistic

# --- Local Environment Setup ---
# Defined prompts help GLIGEN understand context for inpainting
LOCAL_BACKGROUNDS = [
    {"filename": "wooden.jpg", "prompt": "wooden floor"},
    {"filename": "shop1.jpg",   "prompt": "blurred store background"},
    {"filename": "wooden2.jpg",  "prompt": "floor"},
    {"filename": "shop2.jpg", "prompt": "store background"},
    {"filename": "wooden1.jpg", "prompt": "wooden floor"},
]

VERSIONS_PER_IMAGE = 2 # Number of synthetic variants per catalog item

# Ensure project structure exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(LOCAL_BG_DIR):
    os.makedirs(LOCAL_BG_DIR, exist_ok=True)

# --- 1. Load Generative Model (GLIGEN) ---
print("⏳ Loading GLIGEN Model for Grounded Inpainting...")
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-inpainting-text-box",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to("cuda")
pipe.enable_attention_slicing() # Optimize VRAM usage
print("✅ Model Loaded.")

# --- 2. CCTV Augmentation Pipeline ---
# This simulates the physical characteristics of old surveillance footage
cctv_pipeline = A.Compose([
    A.ToGray(p=1.0),                # CCTV is mostly grayscale
    A.Rotate(limit=10, p=0.6),      # Camera angle variations
    A.Perspective(scale=(0.02, 0.07), keep_size=True, p=0.8), # Distortion
    A.GaussNoise(var_limit=(10.0, 27.0), p=0.8), # Digital sensor noise
    A.MotionBlur(blur_limit=5, p=0.3),           # Fast movement blur
    A.RandomBrightnessContrast(p=0.5),           # Unpredictable lighting
])

# --- Helper: Occlusion Logic ---
def get_occlusion_data():
    """
    Generates bounding boxes for hands and stickers to simulate real-world occlusions.
    Uses normalized coordinates [ymin, xmin, ymax, xmax].
    """
    boxes = [
        [0.0, 0.45, 0.25, 0.95], # Hand 1 position
        [0.75, 0.45, 1.0, 0.95]  # Hand 2 position
    ]
    hand_prompt = "a realistic human hand showing distinct fingers gripping the case"
    phrases = [hand_prompt, hand_prompt]
    prompt_parts = ["held by human hands"]

    # Randomly add a price sticker occlusion
    if random.random() < 0.4:
        sticker_type = random.choice(["price tag sticker", "sale label sticker"])
        box_sticker = [0.75, 0.1, 0.95, 0.25] if random.random() > 0.5 else [0.05, 0.1, 0.25, 0.25]
        boxes.append(box_sticker)
        phrases.append(sticker_type)
        prompt_parts.append(f"with a {sticker_type}")

    return {"boxes": boxes, "phrases": phrases, "prompt_suffix": ", ".join(prompt_parts)}

def safe_save(path, img):
    """Handles Windows path encoding issues for OpenCV saving."""
    try:
        is_success, im_buf = cv2.imencode(".jpg", img)
        if is_success: im_buf.tofile(path)
    except Exception as e:
        print(f"❌ Save Error: {e}")

# --- 3. Main Processing Logic ---
def process_single_image(img_name, case_path, backgrounds_list, version_idx):
    clean_name = re.sub(r'[<>:"/\\|?*]', '', os.path.splitext(img_name)[0]).strip()[:50]
    save_path = os.path.join(OUTPUT_DIR, f"{clean_name}_v{version_idx:02d}.jpg")
    
    if os.path.exists(save_path): return

    # Load and prepare album cover
    album_path = os.path.join(CATALOG_DIR, img_name)
    try:
        img_cover = np.array(Image.open(album_path).convert("RGB")) 
    except: return

    # Merge cover with Jewel Case template and background
    img_box = cv2.resize(cv2.imread(case_path), (512, 512))
    bg_data = random.choice(backgrounds_list) if backgrounds_list else {"path": None, "prompt": "neutral"}
    
    img_bg = cv2.imread(bg_data["path"]) if bg_data["path"] else None
    if img_bg is not None:
        img_bg = cv2.resize(img_bg, (512, 512))
        img_merged = (img_bg.astype(float) * img_box.astype(float) / 255).astype(np.uint8)
    else:
        img_merged = img_box

    # Paste cover into the template coordinates
    ax, ay, aw, ah = 50, 10, 390, 390
    imgcover_bgr = cv2.cvtColor(img_cover, cv2.COLOR_RGB2BGR)
    img_merged[ay:ay+ah, ax:ax+aw] = cv2.resize(imgcover_bgr, (aw, ah))
    
    img_pil_input = Image.fromarray(cv2.cvtColor(img_merged, cv2.COLOR_BGR2RGB))

    # --- Run GLIGEN for Grounded Inpainting ---
    # This is the core novelty: generating realistic occlusions in specific areas
    occlusion = get_occlusion_data()
    prompt = f"black and white, cctv footage style, {occlusion['prompt_suffix']}, background is {bg_data['prompt']}"
    
    with torch.autocast("cuda"):
        out_pil = pipe(
            prompt=prompt,
            gligen_phrases=occlusion['phrases'],
            gligen_boxes=occlusion['boxes'],
            gligen_inpaint_image=img_pil_input,
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=30,
            guidance_scale=8.0, 
        ).images[0]

    # Final Crop and CCTV effects
    img_cv_result = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
    final_img_cv = cctv_pipeline(image=img_cv_result)["image"]
    safe_save(save_path, final_img_cv)

# --- Execution ---
if __name__ == "__main__":
    catalog_images = [f for f in os.listdir(CATALOG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    pbar = tqdm(total=len(catalog_images) * VERSIONS_PER_IMAGE, desc="Generating CCTV Dataset")
    
    for img_name in catalog_images:
        for i in range(VERSIONS_PER_IMAGE):
            process_single_image(img_name, CASE_TEMPLATE_PATH, LOCAL_BACKGROUNDS, i)
            pbar.update(1)
    pbar.close()