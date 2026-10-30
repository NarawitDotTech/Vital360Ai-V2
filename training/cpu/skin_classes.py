"""Class set, label mappings and tiers for skin-v1 (fine-grained head; tiers are derived by summing class probabilities).

TIERS marked REVIEW were chosen by the student team/AI, not by a clinician: they need clinician/teacher review before any
real-world use.
"""
import ast

# (class, tier). Order = model output order.
CLASSES = [
    ("melanoma", "red"), ("basal-cell-carcinoma", "red"), ("squamous-cell-carcinoma", "red"),
    ("actinic-keratosis", "amber"), ("nevus", "green"), ("seborrheic-keratosis", "green"),
    ("dermatofibroma", "green"), ("vascular-lesion", "green"),
    # SCIN-only inflammatory / infectious / other (REVIEW)
    ("eczema", "green"), ("contact-dermatitis", "green"), ("urticaria", "green"), ("psoriasis", "amber"),
    ("tinea", "green"), ("acne", "green"), ("folliculitis", "green"), ("insect-bite", "green"),
    ("herpes-zoster", "amber"), ("herpes-simplex", "green"), ("purpura", "amber"),
]
NAMES = [c for c, _ in CLASSES]
TIER_OF = dict(CLASSES)
TIER_NAMES = ["green", "amber", "red"]

HAM = {"mel": "melanoma", "bcc": "basal-cell-carcinoma", "akiec": "actinic-keratosis", "nv": "nevus",
       "bkl": "seborrheic-keratosis", "df": "dermatofibroma", "vasc": "vascular-lesion"}
PAD = {"MEL": "melanoma", "BCC": "basal-cell-carcinoma", "SCC": "squamous-cell-carcinoma", "ACK": "actinic-keratosis",
       "NEV": "nevus", "SEK": "seborrheic-keratosis"}
# Dropped for too few cases after filtering (<40): impetigo (23), keratosis-pilaris (21).
# SCIN dermatologist condition names (top-weighted label) -> class. Near-duplicates are merged.
SCIN = {
    "Melanoma": "melanoma", "Melanoma in situ": "melanoma", "Basal Cell Carcinoma": "basal-cell-carcinoma",
    "SCC/SCCIS": "squamous-cell-carcinoma", "Actinic Keratosis": "actinic-keratosis", "Melanocytic Nevus": "nevus",
    "SK/ISK": "seborrheic-keratosis", "Dermatofibroma": "dermatofibroma", "Hemangioma": "vascular-lesion",
    "Eczema": "eczema", "Acute dermatitis, NOS": "eczema", "Acute and chronic dermatitis": "eczema",
    "Chronic dermatitis, NOS": "eczema", "Infected eczema": "eczema", "Seborrheic Dermatitis": "eczema",
    "Allergic Contact Dermatitis": "contact-dermatitis", "Irritant Contact Dermatitis": "contact-dermatitis",
    "CD - Contact dermatitis": "contact-dermatitis",
    "Urticaria": "urticaria", "Psoriasis": "psoriasis", "Tinea": "tinea", "Tinea Versicolor": "tinea", "Acne": "acne",
    "Folliculitis": "folliculitis", "Insect Bite": "insect-bite", "Herpes Zoster": "herpes-zoster",
    "Herpes Simplex": "herpes-simplex",
    "O/E - ecchymoses present": "purpura", "Purpura": "purpura", "Pigmented purpuric eruption": "purpura",
}
DOMINANCE = 0.5  # SCIN: keep a case only if its top condition weight >= 0.5 and strictly greater than the runner-up


def scin_class(wlabel: str):
    """Weighted-label string -> class name or None (ungradable, ambiguous, or not in the class set)."""
    w = ast.literal_eval(wlabel.replace("\\/", "/"))
    if not w:
        return None
    srt = sorted(w.items(), key=lambda kv: -kv[1])
    if srt[0][1] < DOMINANCE or (len(srt) > 1 and srt[0][1] <= srt[1][1]):
        return None
    return SCIN.get(srt[0][0])
