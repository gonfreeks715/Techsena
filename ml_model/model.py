"""
CargoVision AI — Ensemble Model
Models:
  - backend/models/best1.pt           → YOLOv8 (12 classes)
  - backend/models/best2.pt           → YOLOv8 (5 classes)
  - backend/models/best3.pt           → YOLOv8 (23 classes)
  - backend/models/autoencoder.keras  → Keras autoencoder (combined_loss)
  - backend/models/threshold.npy      → anomaly threshold
"""

import cv2
import numpy as np
import base64
import torch
from ultralytics import YOLO
from pathlib import Path

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    import keras
    TF_AVAILABLE = True
    print(f"  ✅ TensorFlow {tf.__version__} available")
except ImportError:
    TF_AVAILABLE = False
    print("  ⚠️  TensorFlow not installed — run: pip install tensorflow")

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM LOSS FUNCTION
# The autoencoder was trained with a custom 'combined_loss'.
# We register it here so Keras can find it when loading the model.
# This covers the most common combined_loss patterns:
#   MSE + SSIM  OR  MSE + MAE  OR  weighted MSE
# ─────────────────────────────────────────────────────────────────────────────
if TF_AVAILABLE:
    @keras.saving.register_keras_serializable(package="builtins", name="combined_loss")
    def combined_loss(y_true, y_pred):
        """
        Combined reconstruction loss: MSE + MAE
        This matches the most common autoencoder training setup.
        If your friend used a different formula, update this function body.
        """
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return mse + 0.1 * mae

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "backend" / "models"

print("\n" + "="*55)
print("  CargoVision AI — Model Loader")
print("="*55)
print(f"  BASE_DIR          : {BASE_DIR}")
print(f"  MODELS_DIR        : {MODELS_DIR}")
print(f"  MODELS_DIR exists : {MODELS_DIR.exists()}")
if MODELS_DIR.exists():
    print(f"  Files found       : {[f.name for f in MODELS_DIR.iterdir()]}")
print("="*55 + "\n")

YOLO1_PATH       = MODELS_DIR / "best1.pt"
YOLO2_PATH       = MODELS_DIR / "best2.pt"
YOLO3_PATH       = MODELS_DIR / "best3.pt"
AUTOENCODER_PATH = MODELS_DIR / "autoencoder.keras"
THRESHOLD_PATH   = MODELS_DIR / "threshold.npy"
MODEL5_PATH      = MODELS_DIR / "model5.pt"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Covers class names from all 3 YOLO models
THREAT_CLASSES = [
    # best1.pt classes
    "knife", "gun", "wrench", "pliers", "scissors", "hammer",
    # best2.pt classes
    "Gun", "Knife", "Pliers", "Scissors", "Wrench",
    # generic threats
    "weapon", "explosive", "drug", "firearm", "pistol",
    "rifle", "blade", "grenade", "ammunition",
]

# Lower-cased for matching
THREAT_CLASSES_LOWER = [t.lower() for t in THREAT_CLASSES]

RISK_WEIGHTS = {
    "gun":      1.0,  "pistol":   1.0,  "rifle":    1.0,  "firearm":  1.0,
    "knife":    0.85, "blade":    0.80, "weapon":   0.90,
    "explosive":1.0,  "grenade":  1.0,  "ammunition":0.90,
    "drug":     0.95, "drug_package":0.95,
    "wrench":   0.30, "pliers":   0.30, "scissors": 0.30, "hammer":   0.25,
    "liquid":   0.50, "organic_mass":0.60,
    "electronic_device":0.20,
}

DECLARATION_RULES = {
    "electronics": ["laptop","phone","computer","device","electronic_device","tablet"],
    "clothing":    ["fabric","textile","garment","clothes","bags"],
    "food":        ["bottle","cup","bowl","organic"],
    "documents":   ["book","paper","document"],
    "machinery":   ["tool","wrench","machinery","auto parts","bicycle","car"],
}

# ── Label cleanup (fixes typos in best3.pt class names) ──────────────────────
LABEL_FIXES = {
    "---- -----":  "unknown_object",
    "clohes":      "clothes",
    "car weels":   "car wheels",
    "auto parts":  "auto parts",   # already fine, kept for completeness
}

def clean_label(label: str) -> str:
    """Fix known typos and normalise label text."""
    return LABEL_FIXES.get(label, label)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class CargoInspector:

    def __init__(self):
        self.device      = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_models = []
        self.yolo_names  = []
        self.autoencoder = None
        self.ae_threshold= 0.02   # default, overwritten by threshold.npy
        self.model5      = None
        self.model5_type = None

        print(f"[CargoVision] Device: {self.device}\n")
        self._load_yolo_models()
        self._load_autoencoder()
        self._load_model5()
        self._print_summary()

    # ── LOADERS ──────────────────────────────────────────────────────────────

    def _load_yolo_models(self):
        for path, name in [
            (YOLO1_PATH, "YOLO-1 (best1.pt)"),
            (YOLO2_PATH, "YOLO-2 (best2.pt)"),
            (YOLO3_PATH, "YOLO-3 (best3.pt)"),
        ]:
            print(f"  Loading {name} ...")
            if not path.exists():
                print(f"    ❌ Not found: {path}")
                continue
            try:
                model   = YOLO(str(path))
                classes = list(model.names.values())
                self.yolo_models.append(model)
                self.yolo_names.append(name)
                print(f"    ✅ {len(classes)} classes: {classes[:6]}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")

        if not self.yolo_models:
            fp = str(BASE_DIR / "backend" / "yolov8n.pt")
            print(f"  ⚠️  No custom YOLO — fallback: {fp}")
            try:
                self.yolo_models.append(YOLO(fp))
                self.yolo_names.append("YOLO-base (fallback)")
            except Exception as e:
                print(f"  ❌ Fallback failed: {e}")

    def _load_autoencoder(self):
        print(f"\n  Loading Autoencoder (Keras) ...")

        if not TF_AVAILABLE:
            print("    ❌ TensorFlow not available")
            return

        if not AUTOENCODER_PATH.exists():
            print(f"    ❌ autoencoder.keras not found")
            return

        # ── Strategy 1: load with registered combined_loss ────────────────
        try:
            self.autoencoder = keras.models.load_model(
                str(AUTOENCODER_PATH),
                custom_objects={"combined_loss": combined_loss},
            )
            print(f"    ✅ Loaded with custom_objects")
            print(f"       Input  shape: {self.autoencoder.input_shape}")
            print(f"       Output shape: {self.autoencoder.output_shape}")
            self._load_threshold()
            return
        except Exception as e1:
            print(f"    ⚠️  Strategy 1 failed: {e1}")

        # ── Strategy 2: load without compiling (inference only) ───────────
        try:
            self.autoencoder = keras.models.load_model(
                str(AUTOENCODER_PATH),
                custom_objects={"combined_loss": combined_loss},
                compile=False,
            )
            print(f"    ✅ Loaded with compile=False")
            print(f"       Input  shape: {self.autoencoder.input_shape}")
            self._load_threshold()
            return
        except Exception as e2:
            print(f"    ⚠️  Strategy 2 failed: {e2}")

        # ── Strategy 3: safe_mode=False ───────────────────────────────────
        try:
            self.autoencoder = keras.models.load_model(
                str(AUTOENCODER_PATH),
                custom_objects={"combined_loss": combined_loss},
                compile=False,
                safe_mode=False,
            )
            print(f"    ✅ Loaded with safe_mode=False")
            self._load_threshold()
            return
        except Exception as e3:
            print(f"    ❌ All strategies failed")
            print(f"       Last error: {e3}")
            print(f"       Ask your friend what exact combined_loss formula they used")
            self.autoencoder = None

    def _load_threshold(self):
        """Load threshold.npy — called right after autoencoder loads."""
        print(f"  Loading threshold.npy ...")
        if not THRESHOLD_PATH.exists():
            print(f"    ⚠️  Not found — using default: {self.ae_threshold}")
            return
        try:
            data = np.load(str(THRESHOLD_PATH))
            self.ae_threshold = float(data) if data.ndim == 0 else float(data.flat[0])
            print(f"    ✅ Threshold: {self.ae_threshold:.6f}")
        except Exception as e:
            print(f"    ⚠️  Could not load threshold: {e} — using default {self.ae_threshold}")

    def _load_model5(self):
        print(f"\n  Loading Model-5 ...")
        if not MODEL5_PATH.exists():
            print(f"    ℹ️  model5.pt not found — skipping")
            return
        try:
            self.model5      = YOLO(str(MODEL5_PATH))
            self.model5_type = "yolo"
            print(f"    ✅ Model-5 (YOLO) loaded")
        except Exception:
            try:
                self.model5      = torch.load(str(MODEL5_PATH), map_location=self.device)
                self.model5.eval()
                self.model5_type = "torch"
                print(f"    ✅ Model-5 (PyTorch) loaded")
            except Exception as e:
                print(f"    ❌ Model-5 failed: {e}")

    def _print_summary(self):
        print("\n" + "="*55)
        print("  ENSEMBLE READY")
        print("="*55)
        for name in self.yolo_names:
            print(f"  ✅ {name}")
        if self.autoencoder:
            print(f"  ✅ Keras Autoencoder (threshold={self.ae_threshold:.6f})")
        else:
            print(f"  ❌ Autoencoder (see error above)")
        if self.model5:
            print(f"  ✅ Model-5 ({self.model5_type})")
        else:
            print(f"  ❌ Model-5 (not loaded)")
        print(f"  Active : {len(self.yolo_names) + (1 if self.autoencoder else 0)}")
        print("="*55 + "\n")

    # ── PREPROCESSING ─────────────────────────────────────────────────────────

    def _preprocess(self, image_bytes):
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh_gray = clahe.apply(gray)
        enh_bgr  = cv2.cvtColor(enh_gray, cv2.COLOR_GRAY2BGR)
        return img, enh_bgr, enh_gray

    def _preprocess_for_ae(self, gray_img):
        """Auto-detect input shape from loaded Keras model."""
        input_shape = self.autoencoder.input_shape
        h = input_shape[1] or 128
        w = input_shape[2] or 128
        c = input_shape[3] if len(input_shape) > 3 else 1

        resized     = cv2.resize(gray_img, (w, h))
        normalized  = resized.astype(np.float32) / 255.0

        if c == 1:
            return normalized.reshape(1, h, w, 1)
        else:
            rgb = np.stack([normalized] * 3, axis=-1)
            return rgb.reshape(1, h, w, 3)

    # ── YOLO ENSEMBLE ─────────────────────────────────────────────────────────

    def _run_yolo_ensemble(self, img):
        all_dets, model_scores = [], []

        for model, name in zip(self.yolo_models, self.yolo_names):
            try:
                results   = model(img, verbose=False)
                this_dets = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf  = float(box.conf[0])
                        cls   = int(box.cls[0])
                        label = clean_label(model.names.get(cls, f"class_{cls}"))
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if conf < 0.25:
                            continue
                        this_dets.append({
                            "label":      label,
                            "confidence": round(conf, 3),
                            "bbox":       [x1, y1, x2, y2],
                            "source":     name,
                        })
                score = self._score_detections(this_dets)
                model_scores.append(score)
                all_dets.extend(this_dets)
                print(f"    {name}: {len(this_dets)} dets | score={score:.1f}")
            except Exception as e:
                print(f"    ⚠️  {name} error: {e}")

        return self._merge_detections(all_dets), model_scores

    def _merge_detections(self, dets, iou_thr=0.45):
        if not dets:
            return []
        used, merged = [False]*len(dets), []
        for i, a in enumerate(dets):
            if used[i]:
                continue
            group, used[i] = [a], True
            for j, b in enumerate(dets):
                if used[j] or i == j:
                    continue
                if a["label"].lower() == b["label"].lower() and self._iou(a["bbox"], b["bbox"]) > iou_thr:
                    group.append(b)
                    used[j] = True
            avg_conf = sum(d["confidence"] for d in group) / len(group)
            boosted  = min(1.0, avg_conf + 0.05 * (len(group)-1))
            avg_bbox = [int(sum(d["bbox"][k] for d in group)/len(group)) for k in range(4)]
            merged.append({
                "label":       group[0]["label"],
                "confidence":  round(boosted, 3),
                "bbox":        avg_bbox,
                "sources":     list({d["source"] for d in group}),
                "model_count": len(group),
            })
        return sorted(merged, key=lambda x: x["confidence"], reverse=True)

    def _iou(self, a, b):
        xa, ya = max(a[0],b[0]), max(a[1],b[1])
        xb, yb = min(a[2],b[2]), min(a[3],b[3])
        inter  = max(0,xb-xa)*max(0,yb-ya)
        union  = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/union if union > 0 else 0.0

    # ── KERAS AUTOENCODER ─────────────────────────────────────────────────────

    def _run_autoencoder(self, gray_img):
        if self.autoencoder is None:
            return None
        try:
            inp           = self._preprocess_for_ae(gray_img)
            reconstructed = self.autoencoder.predict(inp, verbose=0)
            recon_error   = float(np.mean((inp - reconstructed) ** 2))
            threshold     = self.ae_threshold

            print(f"    Autoencoder: error={recon_error:.6f} | threshold={threshold:.6f}")

            if recon_error <= threshold:
                score = (recon_error / threshold) * 40.0
                print(f"    → NORMAL (score={score:.1f})")
            else:
                ratio = min((recon_error - threshold) / (2 * threshold), 1.0)
                score = 40.0 + ratio * 60.0
                print(f"    → ANOMALY (score={score:.1f})")

            return round(score, 1)
        except Exception as e:
            print(f"    ❌ Autoencoder inference error: {e}")
            return None

    # ── MODEL 5 ───────────────────────────────────────────────────────────────

    def _run_model5(self, img, gray_img):
        if self.model5 is None:
            return None
        try:
            if self.model5_type == "yolo":
                results = self.model5(img, verbose=False)
                dets    = []
                for r in results:
                    if r.boxes is None: continue
                    for box in r.boxes:
                        if float(box.conf[0]) > 0.25:
                            dets.append({
                                "label":      self.model5.names.get(int(box.cls[0]),"unknown"),
                                "confidence": float(box.conf[0]),
                                "bbox":       [],
                            })
                score = self._score_detections(dets)
            else:
                inp   = self._preprocess_for_ae(gray_img)
                t     = torch.tensor(inp).to(self.device)
                with torch.no_grad():
                    out = self.model5(t)
                score = min(100.0, max(0.0, float(out.max().item())*100))
            print(f"    Model-5: score={score:.1f}")
            return round(score, 1)
        except Exception as e:
            print(f"    ⚠️  Model-5 error: {e}")
            return None

    # ── SCORING ───────────────────────────────────────────────────────────────

    def _score_detections(self, dets):
        if not dets:
            return 8.0
        max_risk = 0.0
        for d in dets:
            label = d["label"].lower()
            conf  = d["confidence"]
            for threat, weight in RISK_WEIGHTS.items():
                if threat.lower() in label:
                    max_risk = max(max_risk, weight * conf * 100)
                    break
        if len(dets) > 3:
            max_risk = min(100.0, max_risk + 8.0)
        return round(max_risk, 1)

    def _risk_level(self, score):
        return "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"

    def _check_misdeclaration(self, detections, declared_type):
        expected   = DECLARATION_RULES.get(declared_type.lower(), [])
        mismatches = []
        for d in detections:
            label = d["label"].lower()
            if any(t in label for t in THREAT_CLASSES_LOWER) and not any(e in label for e in expected):
                mismatches.append({
                    "detected": d["label"],
                    "declared": declared_type,
                    "severity": "HIGH",
                })
        return mismatches

    # ── ANNOTATION ────────────────────────────────────────────────────────────

    def _annotate(self, img, detections):
        out = img.copy()
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            label  = det["label"]
            conf   = det["confidence"]
            count  = det.get("model_count", 1)
            is_thr = any(t in label.lower() for t in THREAT_CLASSES_LOWER)
            color  = (0, 0, 255) if is_thr else (0, 200, 80)
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            text = f"{label} {conf:.2f} [{count}M]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(out, text, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return out

    # ── MAIN INSPECT ──────────────────────────────────────────────────────────

    def inspect(self, image_bytes, declared_type="unknown"):
        print(f"\n{'─'*50}")
        print(f"[Inspect] declared_type = {declared_type}")

        _, enhanced, gray = self._preprocess(image_bytes)
        scores_pool       = []

        print("[Inspect] → YOLO ensemble")
        detections, yolo_scores = self._run_yolo_ensemble(enhanced)
        scores_pool.extend(yolo_scores)

        print("[Inspect] → Keras Autoencoder")
        ae_score = self._run_autoencoder(gray)
        if ae_score is not None:
            scores_pool.append(ae_score)

        print("[Inspect] → Model-5")
        m5_score = self._run_model5(enhanced, gray)
        if m5_score is not None:
            scores_pool.append(m5_score)

        final_score = (
            round(sum(scores_pool)/len(scores_pool), 1)
            if scores_pool else self._score_detections(detections)
        )

        mismatches = self._check_misdeclaration(detections, declared_type)
        if mismatches:
            final_score = min(100.0, final_score + 15.0)

        final_score = round(final_score, 1)
        risk_level  = self._risk_level(final_score)

        print(f"\n[Inspect] Scores      : {scores_pool}")
        print(f"[Inspect] Final score : {final_score} → {risk_level}")
        print(f"[Inspect] Detections  : {len(detections)}")
        print(f"[Inspect] Mismatches  : {len(mismatches)}")

        annotated     = self._annotate(enhanced, detections)
        _, buf        = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buf).decode("utf-8")

        breakdown = [{"model": n, "score": s} for n, s in zip(self.yolo_names, yolo_scores)]
        if ae_score is not None:
            breakdown.append({"model": "Autoencoder (Keras)", "score": ae_score})
        if m5_score is not None:
            breakdown.append({"model": "Model-5", "score": m5_score})

        return {
            "risk_score":      final_score,
            "risk_level":      risk_level,
            "detections":      detections,
            "mismatches":      mismatches,
            "total_objects":   len(detections),
            "annotated_image": annotated_b64,
            "model_breakdown": breakdown,
            "models_used":     len(scores_pool),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
inspector = CargoInspector()