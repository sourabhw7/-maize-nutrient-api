import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Maize Nutrient Advisory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "maize_nutrient_model_ft.keras")
print("DEBUG cwd:", os.getcwd())
print("DEBUG BASE_DIR:", BASE_DIR)
print("DEBUG files in BASE_DIR:", os.listdir(BASE_DIR))
print("DEBUG model exists:", os.path.exists(MODEL_PATH),
      "size:", os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else "n/a")
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("DEBUG model loaded OK, input shape:", MODEL.input_shape)

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.60
MAX_BYTES = 5 * 1024 * 1024

# ---- Safeguards against non-leaf photos (tune these after testing) ----
MIN_CONF = 0.70     # top class must have at least 70% probability
MIN_MARGIN = 0.15   # top class must beat the 2nd class by at least 15 points
MIN_PLANT = 0.05    # at least 5% of pixels must be leaf-coloured (yellow-green range)

# Confirmed alphabetical order (matches how Keras sorted the training folders)
CLASSES = ['ALL Present', 'ALLAB', 'KAB', 'NAB', 'PAB', 'ZNAB']

ADVISORY = {
    'ALL Present': {
        'condition': 'Healthy Crop (All Nutrients Present)',
        'fertilizer': 'No Extra Fertilizer Needed',
        'symptoms': 'Leaves exhibit healthy green color and balanced cellular structure.',
        'action': 'Maintain standard irrigation and micro-nutrient maintenance schedule.',
        'stock': 100,
        'price': 'N/A'
    },
    'NAB': {
        'condition': 'Nitrogen Deficiency',
        'fertilizer': 'Urea / Ammonium Sulphate',
        'symptoms': 'General pale green to yellow leaves starting from older bottom leaves.',
        'action': 'Apply top dressing of Urea in split applications.',
        'stock': 45,
        'price': '₹266.50 / 45kg'
    },
    'PAB': {
        'condition': 'Phosphorus Deficiency',
        'fertilizer': 'DAP (Diammonium Phosphate) / SSP',
        'symptoms': 'Purplish tint along leaf edges and stunted root development.',
        'action': 'Apply DAP as a basal placement directly into root zone.',
        'stock': 25,
        'price': '₹1,350.00 / 50kg'
    },
    'KAB': {
        'condition': 'Potassium Deficiency',
        'fertilizer': 'MOP (Muriate of Potash)',
        'symptoms': 'Marginal leaf scorch, browning edges, and weak stem strength.',
        'action': 'Apply MOP to restore water regulation and plant immunity.',
        'stock': 12,
        'price': '₹1,700.00 / 50kg'
    },
    'ZNAB': {
        'condition': 'Zinc Deficiency',
        'fertilizer': 'Zinc Sulphate (ZnSO4)',
        'symptoms': 'Broad white/bleached bands on leaf tissues beside the midrib.',
        'action': 'Foliar spray with 0.5% Zinc Sulphate solution mixed with slaked lime.',
        'stock': 30,
        'price': '₹550.00 / 25kg'
    },
    'ALLAB': {
        'condition': 'Severe Multiple Deficiencies',
        'fertilizer': 'NPK Complex (19:19:19) + Micronutrient Mix',
        'symptoms': 'Multi-pattern chlorosis, severe necrosis, and severely stunted growth.',
        'action': 'Apply balanced water-soluble NPK foliar spray. Soil test recommended.',
        'stock': 20,
        'price': '₹1,200.00 / 25kg'
    }
}


def load_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img.resize(IMG_SIZE, Image.BILINEAR)


def to_tensor(img: Image.Image) -> np.ndarray:
    # Matches the training pipeline exactly: PIL, RGB, bilinear resize, raw 0-255 pixels.
    # Preprocessing (mobilenet_v2 preprocess_input) is baked INSIDE the model itself,
    # so we must NOT normalize or divide by 255 here.
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def plant_ratio(img: Image.Image) -> float:
    """Fraction of pixels in the yellow-to-green colour range (leaf-like)."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    h = hsv[..., 0] * 360.0 / 255.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    mask = (h >= 30) & (h <= 170) & (s >= 0.20) & (v >= 0.15)
    return float(mask.mean())


def reject_reason(probs: np.ndarray, ratio: float):
    order = np.argsort(probs)[::-1]
    top1, top2 = float(probs[order[0]]), float(probs[order[1]])
    if ratio < MIN_PLANT:
        return "no_plant"
    if top1 < MIN_CONF:
        return "low_conf"
    if top1 - top2 < MIN_MARGIN:
        return "ambiguous"
    return None


@app.get("/")
def home():
    return {"status": "ok", "mode": "real", "classes": CLASSES}


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Please upload a JPG or PNG image.")

    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(413, "Image is too large (max 5 MB).")

    try:
        img = load_image(contents)
        tensor = to_tensor(img)
    except Exception:
        raise HTTPException(400, "Could not read this image.")

    probs = MODEL(tensor, training=False).numpy()[0]
    order = np.argsort(probs)[::-1]
    best_idx = int(order[0])
    label = CLASSES[best_idx]
    confidence = float(probs[best_idx])

    # Safeguard: reject photos that do not look like a clear maize leaf
    ratio = plant_ratio(img)
    reason = reject_reason(probs, ratio)
    print(f"DEBUG predict: label={label} conf={confidence:.3f} "
          f"plant_ratio={ratio:.3f} reject={reason}")
    if reason:
        return {
            "not_leaf": True,
            "low_confidence": True,
            "reason": reason,
            "message": "Could not recognise a clear maize leaf. Please retake the photo "
                       "in good light with the leaf filling the frame."
        }

    top3 = [
        {"label": CLASSES[int(i)], "confidence": round(float(probs[int(i)]), 4)}
        for i in order[:3]
    ]

    info = ADVISORY[label]

    return {
        "detected_condition": info["condition"],
        "confidence": round(confidence * 100, 1),
        "recommended_fertilizer": info["fertilizer"],
        "symptoms": info["symptoms"],
        "action": info["action"],
        "stock": {
            "in_stock": info["stock"] > 0,
            "quantity_bags": info["stock"],
            "unit_price": info["price"]
        },
        "low_confidence": confidence < CONF_THRESHOLD,
        "top3": top3
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
