from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import joblib
import nibabel as nib
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- Model slots (None = unavailable) ---
tf_binary_model: tf.keras.Model | None = None
tf_multi_model: tf.keras.Model | None = None
pt_binary_model: nn.Module | None = None
pt_multi_model: nn.Module | None = None
hybrid_binary_extractor: Any | None = None
hybrid_binary_clf: nn.Module | None = None
hybrid_multi_extractor: Any | None = None
hybrid_multi_clf: nn.Module | None = None
# Full sklearn Pipeline (e.g. scaler + PCA + SVM) — no separate PyTorch head
hybrid_binary_sklearn: Any | None = None
hybrid_multi_sklearn: Any | None = None

LABEL_ORDER: tuple[Literal["AD", "MCI", "CN"], ...] = ("AD", "MCI", "CN")


def resolve_model_path(env_key: str, default: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    val = os.getenv(env_key, default)
    return val if os.path.isabs(val) else os.path.join(base, val)


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    parts = [o.strip() for o in raw.split(",") if o.strip()]
    return parts if parts else ["http://localhost:3000"]


TEMP_DIR = os.getenv("TEMP_DIR", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def _load_volume_array_from_scan(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            raise ValueError(".npy must contain a numpy ndarray")
        vol = np.asarray(arr, dtype=np.float32)
        if vol.ndim == 4:
            vol = vol[..., 0]
        if vol.ndim == 2:
            vol = vol[:, :, np.newaxis]
        if vol.ndim != 3:
            raise ValueError(f"Expected 2D–4D array for volume, got {vol.shape}")
        return vol
    nii = nib.load(path)
    vol = np.asanyarray(nii.dataobj, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D (or 4D) volume, got shape {vol.shape}")
    return vol


def _volume_to_slice_hwc(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim == 2:
        sl = vol
    elif vol.ndim == 3:
        k = vol.shape[2] // 2
        sl = vol[:, :, k]
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D volume, got shape {vol.shape}")
    smin = float(sl.min())
    smax = float(sl.max())
    if smax > smin:
        sl = (sl - smin) / (smax - smin)
    else:
        sl = np.zeros_like(sl)
    t = tf.constant(sl[..., np.newaxis], dtype=tf.float32)
    t = tf.image.resize(t, [128, 128])
    t = tf.repeat(t, 3, axis=-1)
    return t.numpy()


def nifti_to_binary_input(path: str) -> np.ndarray:
    """TF preprocessing: NIfTI / .npy volume → batch (1, 128, 128, 3)."""
    vol = _load_volume_array_from_scan(path)
    hwc = _volume_to_slice_hwc(vol)
    return np.expand_dims(hwc, axis=0)


def nifti_to_pt_input(path: str) -> torch.Tensor:
    """PyTorch preprocessing: same slice as TF, shape (1, 3, 128, 128)."""
    batch_hwc = nifti_to_binary_input(path)
    x = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).contiguous().float()
    return x


# --- ResNet18 / Custom CNN slot (matches Collab 2D CNN notebook) ---
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _norm01_2d(sl: np.ndarray) -> np.ndarray:
    smin = float(sl.min())
    smax = float(sl.max())
    if smax > smin:
        return (sl - smin) / (smax - smin)
    return np.zeros_like(sl, dtype=np.float32)


def _resize_slice_hw(sl: np.ndarray, size: int = 128) -> np.ndarray:
    t = tf.constant(sl[..., np.newaxis], dtype=tf.float32)
    t = tf.image.resize(t, [size, size])
    return t.numpy()[..., 0].astype(np.float32)


def _volume_to_25d_chw(vol: np.ndarray, offset: int = 5) -> np.ndarray:
    """Three axial slices as RGB-like channels, 128×128 (2.5D per training notebook)."""
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim == 2:
        sl = _norm01_2d(vol)
        sl = _resize_slice_hw(sl)
        return np.stack([sl, sl, sl], axis=0)
    if vol.ndim != 3:
        raise ValueError(f"Expected 2D or 3D volume, got shape {vol.shape}")
    d = vol.shape[2]
    if d < 3:
        sl = _norm01_2d(vol[:, :, 0])
        sl = _resize_slice_hw(sl)
        return np.stack([sl, sl, sl], axis=0)
    mid = d // 2
    off = min(offset, max(1, (d - 1) // 2 - 1))
    mid = max(off, min(mid, d - off - 1))
    slices = [
        _norm01_2d(vol[:, :, mid - off]),
        _norm01_2d(vol[:, :, mid]),
        _norm01_2d(vol[:, :, mid + off]),
    ]
    resized = [_resize_slice_hw(s) for s in slices]
    return np.stack(resized, axis=0).astype(np.float32)


def nifti_to_resnet_pt_input(path: str) -> torch.Tensor:
    """ResNet18 path: 2.5D slices + ImageNet normalization, shape (1, 3, 128, 128)."""
    vol = _load_volume_array_from_scan(path)
    chw = _volume_to_25d_chw(vol)
    t = torch.from_numpy(chw).float()
    t = torch.clamp(t, 0.0, 1.0)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=t.dtype).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)


def _confidence_from_binary_probs(p_ad: float) -> dict[str, float]:
    p_ad = float(np.clip(p_ad, 0.0, 1.0))
    p_cn = 1.0 - p_ad
    return {
        "AD": round(p_ad * 100.0, 2),
        "MCI": 0.0,
        "CN": round(p_cn * 100.0, 2),
    }


def _confidence_from_multiclass_probs(p: np.ndarray) -> dict[str, float]:
    p = np.asarray(p, dtype=np.float64).flatten()
    if p.size != 3:
        raise ValueError(f"Expected 3 class probabilities, got shape {p.shape}")
    p = np.clip(p, 0.0, 1.0)
    s = p.sum()
    if s > 0:
        p = p / s
    return {
        "AD": round(float(p[0]) * 100.0, 2),
        "MCI": round(float(p[1]) * 100.0, 2),
        "CN": round(float(p[2]) * 100.0, 2),
    }


def _prediction_from_multiclass_probs(p: np.ndarray) -> Literal["AD", "MCI", "CN"]:
    idx = int(np.argmax(np.asarray(p).flatten()))
    return LABEL_ORDER[idx]


def sync_multiclass_mock() -> dict[str, Any]:
    time.sleep(2.5)
    predicted_class = random.choice(["AD", "MCI", "CN"])
    if predicted_class == "AD":
        confidence = {"AD": 82.1, "MCI": 12.4, "CN": 5.5}
    elif predicted_class == "MCI":
        confidence = {"AD": 15.3, "MCI": 75.2, "CN": 9.5}
    else:
        confidence = {"AD": 2.1, "MCI": 10.4, "CN": 87.5}
    return {"prediction": predicted_class, "confidence": confidence}


def _load_torch_module(path: str) -> nn.Module:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj
    raise TypeError(f"Expected nn.Module in {path}, got {type(obj)}")


def _try_load_tf(path: str, label: str) -> tf.keras.Model | None:
    if not os.path.isfile(path):
        logger.warning(
            "WARNING: [%s] model not found at %s — will be skipped during inference.",
            label,
            path,
        )
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        logger.warning(
            "WARNING: [%s] failed to load from %s — %s",
            label,
            path,
            e,
        )
        return None


def _try_load_torch(path: str, label: str) -> nn.Module | None:
    if not os.path.isfile(path):
        logger.warning(
            "WARNING: [%s] model not found at %s — will be skipped during inference.",
            label,
            path,
        )
        return None
    try:
        return _load_torch_module(path)
    except Exception as e:
        logger.warning(
            "WARNING: [%s] failed to load from %s — %s",
            label,
            path,
            e,
        )
        return None


def _try_load_pt_classifier(path: str, label: str) -> nn.Module | None:
    """Load PyTorch Custom CNN slot: full nn.Module or ResNet18 ``state_dict`` (``.pth``)."""
    if not os.path.isfile(path):
        logger.warning(
            "WARNING: [%s] model not found at %s — will be skipped during inference.",
            label,
            path,
        )
        return None
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            obj.eval()
            logger.info("Loaded %s as torch.nn.Module from %s", label, path)
            return obj
        if isinstance(obj, dict) and "fc.weight" in obj and "conv1.weight" in obj:
            from torchvision.models import resnet18

            n_classes = int(obj["fc.weight"].shape[0])
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            model.load_state_dict(obj, strict=True)
            model.eval()
            logger.info(
                "Loaded %s as ResNet18 (%d classes) from state_dict %s",
                label,
                n_classes,
                path,
            )
            return model
    except Exception as e:
        logger.warning(
            "WARNING: [%s] failed to load from %s — %s",
            label,
            path,
            e,
        )
    return None


def _is_resnet18_classifier(m: nn.Module) -> bool:
    fc = getattr(m, "fc", None)
    return isinstance(fc, nn.Linear) and hasattr(m, "conv1")


def _try_load_sklearn(path: str, label: str) -> Any | None:
    if not os.path.isfile(path):
        logger.warning(
            "WARNING: [%s] model not found at %s — will be skipped during inference.",
            label,
            path,
        )
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        logger.warning(
            "WARNING: [%s] failed to load from %s — %s",
            label,
            path,
            e,
        )
        return None


def _is_sklearn_classifier_pipeline(obj: Any) -> bool:
    """True if obj is a sklearn Pipeline whose last step can classify (e.g. PCA+SVM)."""
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:
        return False
    if not isinstance(obj, Pipeline) or not getattr(obj, "steps", None):
        return False
    last = obj.steps[-1][1]
    return hasattr(last, "predict_proba") or hasattr(last, "decision_function")


def load_all_models() -> None:
    global tf_binary_model, tf_multi_model
    global pt_binary_model, pt_multi_model
    global hybrid_binary_extractor, hybrid_binary_clf
    global hybrid_multi_extractor, hybrid_multi_clf
    global hybrid_binary_sklearn, hybrid_multi_sklearn

    tf_binary_model = _try_load_tf(
        resolve_model_path("TF_BINARY_MODEL_PATH", "models/tf/task2_binary_model.h5"),
        "TF binary",
    )
    tf_multi_model = _try_load_tf(
        resolve_model_path("TF_MULTI_MODEL_PATH", "models/tf/task2_multi_model.h5"),
        "TF multiclass",
    )

    pt_binary_model = _try_load_pt_classifier(
        resolve_model_path("PT_BINARY_MODEL_PATH", "models/pt/task2_binary_model.pth"),
        "PT binary",
    )
    pt_multi_model = _try_load_pt_classifier(
        resolve_model_path("PT_MULTI_MODEL_PATH", "models/pt/task2_multi_model.pth"),
        "PT multiclass",
    )

    hb_pkl = resolve_model_path(
        "HYBRID_BINARY_PKL_PATH",
        "models/hybrid/task2_binary_features.pkl",
    )
    hb_clf = resolve_model_path(
        "HYBRID_BINARY_CLF_PATH",
        "models/hybrid/task2_binary_clf.pth",
    )
    hm_pkl = resolve_model_path(
        "HYBRID_MULTI_PKL_PATH",
        "models/hybrid/task2_multi_features.pkl",
    )
    hm_clf = resolve_model_path(
        "HYBRID_MULTI_CLF_PATH",
        "models/hybrid/task2_multi_clf.pth",
    )

    hybrid_binary_sklearn = None
    hybrid_multi_sklearn = None
    hybrid_binary_extractor = None
    hybrid_binary_clf = None
    hybrid_multi_extractor = None
    hybrid_multi_clf = None

    hb_pkl_obj = _try_load_sklearn(hb_pkl, "Hybrid binary pkl")
    if hb_pkl_obj is not None and _is_sklearn_classifier_pipeline(hb_pkl_obj):
        hybrid_binary_sklearn = hb_pkl_obj
        logger.info("Hybrid binary: sklearn Pipeline (e.g. PCA+SVM); PyTorch head not required.")
    elif hb_pkl_obj is not None:
        hybrid_binary_extractor = hb_pkl_obj
        hybrid_binary_clf = _try_load_torch(hb_clf, "Hybrid binary clf")
        if hybrid_binary_extractor is None or hybrid_binary_clf is None:
            hybrid_binary_extractor = None
            hybrid_binary_clf = None

    hm_pkl_obj = _try_load_sklearn(hm_pkl, "Hybrid multiclass pkl")
    if hm_pkl_obj is not None and _is_sklearn_classifier_pipeline(hm_pkl_obj):
        hybrid_multi_sklearn = hm_pkl_obj
        logger.info(
            "Hybrid multiclass: sklearn Pipeline (e.g. PCA+SVM); PyTorch head not required.",
        )
    elif hm_pkl_obj is not None:
        hybrid_multi_extractor = hm_pkl_obj
        hybrid_multi_clf = _try_load_torch(hm_clf, "Hybrid multiclass clf")
        if hybrid_multi_extractor is None or hybrid_multi_clf is None:
            hybrid_multi_extractor = None
            hybrid_multi_clf = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


app = FastAPI(title="CerebroAI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _hybrid_features(slice_hwc: np.ndarray, extractor: Any) -> np.ndarray:
    flat = slice_hwc.astype(np.float32).reshape(1, -1)
    if hasattr(extractor, "transform"):
        return np.asarray(extractor.transform(flat), dtype=np.float32)
    if hasattr(extractor, "predict") and not hasattr(extractor, "transform"):
        return np.asarray(extractor.predict(flat), dtype=np.float32)
    raise TypeError("Hybrid .pkl object must support .transform()")


def _torch_logits_to_binary_result(logits: torch.Tensor) -> dict[str, Any]:
    v = logits[0] if logits.dim() == 2 else logits.flatten()
    if v.numel() == 2:
        # CrossEntropy with two classes (e.g. ResNet): index 1 = AD in training notebook
        p = torch.softmax(v, dim=0).detach().cpu().numpy()
        p_ad = float(p[1])
    elif v.numel() == 1:
        p_ad = torch.sigmoid(v).item()
    else:
        p_ad = torch.sigmoid(v.flatten()[0]).item()
    pred = "AD" if p_ad >= 0.5 else "CN"
    return {"prediction": pred, "confidence": _confidence_from_binary_probs(p_ad)}


def _pt_resnet_multiclass_logits_to_result(logits: torch.Tensor) -> dict[str, Any]:
    """Training used class order CN=0, MCI=1, AD=2; API uses AD, MCI, CN indices."""
    v = logits[0] if logits.dim() == 2 else logits.flatten()
    if v.numel() != 3:
        raise ValueError(f"Expected 3-class logits, got shape {tuple(logits.shape)}")
    p_cma = torch.softmax(v, dim=0).detach().cpu().numpy()
    probs = np.array([p_cma[2], p_cma[1], p_cma[0]], dtype=np.float64)
    pred = _prediction_from_multiclass_probs(probs)
    return {"prediction": pred, "confidence": _confidence_from_multiclass_probs(probs)}


def _torch_logits_to_multiclass_result(logits: torch.Tensor) -> dict[str, Any]:
    if logits.dim() == 2:
        v = logits[0]
    else:
        v = logits.flatten()
    if v.numel() == 1:
        p = torch.sigmoid(v).item()
        probs = np.array([p, 0.0, 1.0 - p], dtype=np.float64)
    elif v.numel() == 3:
        probs = torch.softmax(v, dim=0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected multiclass logits shape {tuple(logits.shape)}")
    pred = _prediction_from_multiclass_probs(probs)
    return {"prediction": pred, "confidence": _confidence_from_multiclass_probs(probs)}


def run_tensorflow_binary(path: str) -> dict[str, Any]:
    if tf_binary_model is None:
        raise RuntimeError("TF binary unavailable")
    batch = nifti_to_binary_input(path)
    probs = tf_binary_model.predict(batch, verbose=0)
    p_ad = float(np.clip(probs[0, 0], 0.0, 1.0))
    pred = "AD" if p_ad >= 0.5 else "CN"
    return {"prediction": pred, "confidence": _confidence_from_binary_probs(p_ad)}


def run_tensorflow_multiclass(path: str) -> dict[str, Any]:
    if tf_multi_model is None:
        return sync_multiclass_mock()
    batch = nifti_to_binary_input(path)
    out = tf_multi_model.predict(batch, verbose=0)[0]
    out = np.asarray(out, dtype=np.float64).flatten()
    if out.size == 1:
        p_ad = float(np.clip(out[0], 0.0, 1.0))
        probs = np.array([p_ad, 0.0, 1.0 - p_ad], dtype=np.float64)
    else:
        ex = np.exp(out - np.max(out))
        probs = ex / ex.sum()
    pred = _prediction_from_multiclass_probs(probs)
    return {"prediction": pred, "confidence": _confidence_from_multiclass_probs(probs)}


def run_tensorflow_multiclass_strict(path: str) -> dict[str, Any]:
    if tf_multi_model is None:
        raise RuntimeError("TensorFlow multiclass model not loaded")
    batch = nifti_to_binary_input(path)
    out = tf_multi_model.predict(batch, verbose=0)[0]
    out = np.asarray(out, dtype=np.float64).flatten()
    if out.size == 1:
        p_ad = float(np.clip(out[0], 0.0, 1.0))
        probs = np.array([p_ad, 0.0, 1.0 - p_ad], dtype=np.float64)
    else:
        ex = np.exp(out - np.max(out))
        probs = ex / ex.sum()
    pred = _prediction_from_multiclass_probs(probs)
    return {"prediction": pred, "confidence": _confidence_from_multiclass_probs(probs)}


def run_pytorch_binary(path: str) -> dict[str, Any]:
    if pt_binary_model is None:
        raise RuntimeError("PT binary unavailable")
    x = (
        nifti_to_resnet_pt_input(path)
        if _is_resnet18_classifier(pt_binary_model)
        else nifti_to_pt_input(path)
    )
    with torch.no_grad():
        logits = pt_binary_model(x)
    return _torch_logits_to_binary_result(logits)


def run_pytorch_multiclass(path: str) -> dict[str, Any]:
    if pt_multi_model is None:
        return sync_multiclass_mock()
    x = (
        nifti_to_resnet_pt_input(path)
        if _is_resnet18_classifier(pt_multi_model)
        else nifti_to_pt_input(path)
    )
    with torch.no_grad():
        logits = pt_multi_model(x)
    if _is_resnet18_classifier(pt_multi_model):
        return _pt_resnet_multiclass_logits_to_result(logits)
    return _torch_logits_to_multiclass_result(logits)


def run_pytorch_multiclass_strict(path: str) -> dict[str, Any]:
    if pt_multi_model is None:
        raise RuntimeError("PyTorch multiclass model not loaded")
    x = (
        nifti_to_resnet_pt_input(path)
        if _is_resnet18_classifier(pt_multi_model)
        else nifti_to_pt_input(path)
    )
    with torch.no_grad():
        logits = pt_multi_model(x)
    if _is_resnet18_classifier(pt_multi_model):
        return _pt_resnet_multiclass_logits_to_result(logits)
    return _torch_logits_to_multiclass_result(logits)


def _sklearn_pipeline_n_features_in(pipeline: Any) -> int:
    for _, est in pipeline.steps:
        n = getattr(est, "n_features_in_", None)
        if n is not None:
            return int(n)
    raise ValueError("Could not read n_features_in_ from sklearn hybrid pipeline")


def _sklearn_hybrid_feature_matrix(path: str, pipeline: Any) -> np.ndarray:
    """Build the flat feature vector the fitted Pipeline expects (matches training)."""
    from scipy.ndimage import zoom

    n = _sklearn_pipeline_n_features_in(pipeline)
    vol = _load_volume_array_from_scan(path)
    # 128 x 128 x 3 — same as CNN / TF slice path
    if n == 128 * 128 * 3:
        slice_hwc = _volume_to_slice_hwc(vol)
        return slice_hwc.astype(np.float32).reshape(1, -1)
    # Common 3D PCA path: 8^3 = 512 voxels
    if n == 512:
        v = np.asarray(vol, dtype=np.float32)
        if v.ndim == 4:
            v = v[..., 0]
        if v.ndim != 3:
            raise ValueError(f"Expected 3D volume for 512-feature model, got shape {v.shape}")
        target = (8, 8, 8)
        zf = tuple(t / s for t, s in zip(target, v.shape, strict=True))
        resized = zoom(v, zf, order=1)
        return resized.reshape(1, -1)
    # Any perfect cube (e.g. 10^3 = 1000)
    k = round(n ** (1 / 3))
    if k > 0 and k * k * k == n:
        v = np.asarray(vol, dtype=np.float32)
        if v.ndim == 4:
            v = v[..., 0]
        if v.ndim != 3:
            raise ValueError(f"Expected 3D volume for {n}-feature cube model, got {v.shape}")
        target = (k, k, k)
        zf = tuple(t / s for t, s in zip(target, v.shape, strict=True))
        resized = zoom(v, zf, order=1)
        return resized.reshape(1, -1)
    raise ValueError(
        f"Unsupported sklearn hybrid input size n={n}. "
        "Use 49152 (128×128×3 slice) or a perfect cube (e.g. 512=8³).",
    )


def _sklearn_pipeline_binary_result(path: str, pipeline: Any) -> dict[str, Any]:
    """Binary AD/CN from a fitted sklearn Pipeline (e.g. scaler + PCA + SVC)."""
    X = _sklearn_hybrid_feature_matrix(path, pipeline)
    proba = np.asarray(pipeline.predict_proba(X)[0], dtype=np.float64)
    classes = np.asarray(pipeline.classes_)
    # Integer label in training that corresponds to AD (default 1)
    ad_label = int(os.getenv("HYBRID_BINARY_AD_LABEL", "1"))
    if ad_label not in classes:
        ad_label = int(classes.max())
    p_ad = float(np.clip(proba[np.where(classes == ad_label)[0][0]], 0.0, 1.0))
    pred_idx = int(np.argmax(proba))
    pred_label = int(classes[pred_idx])
    pred = "AD" if pred_label == ad_label else "CN"
    return {"prediction": pred, "confidence": _confidence_from_binary_probs(p_ad)}


def _sklearn_pipeline_multiclass_result(path: str, pipeline: Any) -> dict[str, Any]:
    """3-class from sklearn Pipeline; map sklearn class order via HYBRID_MULTI_LABEL_ORDER."""
    X = _sklearn_hybrid_feature_matrix(path, pipeline)
    p = np.asarray(pipeline.predict_proba(X)[0], dtype=np.float64)
    raw = os.getenv("HYBRID_MULTI_LABEL_ORDER", "AD,MCI,CN").strip()
    order = [s.strip() for s in raw.split(",") if s.strip()]
    if len(order) != 3:
        order = ["AD", "MCI", "CN"]
    n = min(len(p), len(order))
    pred_name = order[int(np.argmax(p[:n]))]
    # Build probs in LABEL_ORDER (AD, MCI, CN)
    vec = np.zeros(3, dtype=np.float64)
    for j in range(n):
        name = order[j]
        if name == "AD":
            vec[0] = p[j]
        elif name == "MCI":
            vec[1] = p[j]
        elif name == "CN":
            vec[2] = p[j]
    s = float(vec.sum())
    if s > 0:
        vec = vec / s
    else:
        vec[:] = 1.0 / 3.0
    pred = pred_name if pred_name in LABEL_ORDER else _prediction_from_multiclass_probs(vec)
    return {"prediction": pred, "confidence": _confidence_from_multiclass_probs(vec)}


def run_hybrid_binary(path: str) -> dict[str, Any]:
    if hybrid_binary_sklearn is not None:
        return _sklearn_pipeline_binary_result(path, hybrid_binary_sklearn)
    if hybrid_binary_extractor is None or hybrid_binary_clf is None:
        raise RuntimeError("Hybrid binary unavailable")
    vol = _load_volume_array_from_scan(path)
    slice_hwc = _volume_to_slice_hwc(vol)
    feats = _hybrid_features(slice_hwc, hybrid_binary_extractor)
    t = torch.from_numpy(feats).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    with torch.no_grad():
        logits = hybrid_binary_clf(t)
    return _torch_logits_to_binary_result(logits)


def run_hybrid_multiclass(path: str) -> dict[str, Any]:
    if hybrid_multi_sklearn is not None:
        return _sklearn_pipeline_multiclass_result(path, hybrid_multi_sklearn)
    if hybrid_multi_extractor is None or hybrid_multi_clf is None:
        return sync_multiclass_mock()
    vol = _load_volume_array_from_scan(path)
    slice_hwc = _volume_to_slice_hwc(vol)
    feats = _hybrid_features(slice_hwc, hybrid_multi_extractor)
    t = torch.from_numpy(feats).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    with torch.no_grad():
        logits = hybrid_multi_clf(t)
    return _torch_logits_to_multiclass_result(logits)


def run_hybrid_multiclass_strict(path: str) -> dict[str, Any]:
    if hybrid_multi_sklearn is not None:
        return _sklearn_pipeline_multiclass_result(path, hybrid_multi_sklearn)
    if hybrid_multi_extractor is None or hybrid_multi_clf is None:
        raise RuntimeError("Hybrid multiclass models not loaded")
    vol = _load_volume_array_from_scan(path)
    slice_hwc = _volume_to_slice_hwc(vol)
    feats = _hybrid_features(slice_hwc, hybrid_multi_extractor)
    t = torch.from_numpy(feats).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    with torch.no_grad():
        logits = hybrid_multi_clf(t)
    return _torch_logits_to_multiclass_result(logits)


def _internal_to_api_binary(r: dict[str, Any]) -> dict[str, Any]:
    pred = str(r["prediction"])
    c = r["confidence"]
    ad = float(c.get("AD", 0))
    cn = float(c.get("CN", 0))
    s = ad + cn
    if s > 0:
        ad = ad / s * 100.0
        cn = cn / s * 100.0
    else:
        ad = cn = 50.0
    ad = round(ad, 2)
    cn = round(100.0 - ad, 2)
    res = "ad" if pred == "AD" else "cn"
    return {"result": res, "confidence": {"ad": ad, "cn": cn}}


def _internal_to_api_multiclass(r: dict[str, Any]) -> dict[str, Any]:
    pred = str(r["prediction"])
    c = r["confidence"]
    v = np.array(
        [c.get("AD", 0), c.get("MCI", 0), c.get("CN", 0)],
        dtype=np.float64,
    )
    s = float(v.sum())
    if s > 0:
        v = v / s * 100.0
    else:
        v = np.array([100.0 / 3] * 3)
    ad = round(float(v[0]), 2)
    mci = round(float(v[1]), 2)
    cn = round(100.0 - ad - mci, 2)
    res = pred.lower()
    return {"result": res, "confidence": {"ad": ad, "mci": mci, "cn": cn}}


def _average_binary_internals(ok: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ok:
        return None
    mean_ad = sum(x["confidence"]["AD"] for x in ok) / len(ok)
    mean_cn = sum(x["confidence"]["CN"] for x in ok) / len(ok)
    s = mean_ad + mean_cn
    if s > 0:
        mean_ad = mean_ad / s * 100.0
        mean_cn = mean_cn / s * 100.0
    else:
        mean_ad = mean_cn = 50.0
    ad = round(mean_ad, 2)
    cn = round(100.0 - ad, 2)
    res = "ad" if ad >= cn else "cn"
    return {"result": res, "confidence": {"ad": ad, "cn": cn}}


def _average_multiclass_internals(ok: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ok:
        return None
    ma = sum(x["confidence"]["AD"] for x in ok) / len(ok)
    mm = sum(x["confidence"]["MCI"] for x in ok) / len(ok)
    mc = sum(x["confidence"]["CN"] for x in ok) / len(ok)
    v = np.array([ma, mm, mc], dtype=np.float64)
    s = float(v.sum())
    if s > 0:
        v = v / s * 100.0
    else:
        v = np.array([100.0 / 3] * 3)
    ad = round(float(v[0]), 2)
    mci = round(float(v[1]), 2)
    cn = round(100.0 - ad - mci, 2)
    idx = int(np.argmax([ma, mm, mc]))
    res = ("ad", "mci", "cn")[idx]
    return {"result": res, "confidence": {"ad": ad, "mci": mci, "cn": cn}}


async def run_full_analysis(path: str) -> tuple[dict[str, Any], list[str]]:
    """Run all three pipelines (binary + multiclass each); collect errors per slot."""
    model_errors: list[str] = []
    order = ["2d_cnn_model", "custom_cnn", "3d_pca_svm"]
    runners: list[tuple[Any, Any]] = [
        (run_tensorflow_binary, run_tensorflow_multiclass_strict),
        (run_pytorch_binary, run_pytorch_multiclass_strict),
        (run_hybrid_binary, run_hybrid_multiclass_strict),
    ]

    async def safe(key: str, mode: str, fn: Any) -> dict[str, Any] | None:
        try:
            return await asyncio.to_thread(fn, path)
        except Exception as e:
            model_errors.append(f"{key}.{mode}: {e!s}")
            logger.exception("%s %s failed", key, mode)
            return None

    bin_ok: list[dict[str, Any]] = []
    multi_ok: list[dict[str, Any]] = []
    per_model: dict[str, dict[str, Any]] = {}

    for key, (fb, fm) in zip(order, runners, strict=True):
        b, m = await asyncio.gather(
            safe(key, "binary", fb),
            safe(key, "multiclass", fm),
        )
        if b is not None:
            bin_ok.append(b)
        if m is not None:
            multi_ok.append(m)
        per_model[key] = {
            "binary": _internal_to_api_binary(b) if b is not None else None,
            "multiclass": _internal_to_api_multiclass(m) if m is not None else None,
        }

    body: dict[str, Any] = {
        "averaged": {
            "binary": _average_binary_internals(bin_ok),
            "multiclass": _average_multiclass_internals(multi_ok),
        },
        "3d_pca_svm": per_model["3d_pca_svm"],
        "2d_cnn_model": per_model["2d_cnn_model"],
        "custom_cnn": per_model["custom_cnn"],
    }
    return body, model_errors


def health_models_status() -> dict[str, dict[str, bool]]:
    return {
        "tensorflow": {
            "binary": tf_binary_model is not None,
            "multiclass": tf_multi_model is not None,
        },
        "pytorch": {
            "binary": pt_binary_model is not None,
            "multiclass": pt_multi_model is not None,
        },
        "hybrid": {
            "binary": hybrid_binary_sklearn is not None
            or (
                hybrid_binary_extractor is not None and hybrid_binary_clf is not None
            ),
            "multiclass": hybrid_multi_sklearn is not None
            or (
                hybrid_multi_extractor is not None and hybrid_multi_clf is not None
            ),
        },
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"models": health_models_status()}


def _allowed_upload(name: str) -> bool:
    n = name.lower()
    return n.endswith(".nii.gz") or n.endswith(".nii") or n.endswith(".npy")


@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        if not _allowed_upload(file.filename or ""):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Allowed: .nii, .nii.gz, .npy",
            )

        file_location = os.path.join(TEMP_DIR, file.filename)
        with open(file_location, "wb+") as fo:
            fo.write(await file.read())

        print(f"Received {file.filename} for full analysis...")

        try:
            body, model_errors = await run_full_analysis(file_location)
        finally:
            if os.path.exists(file_location):
                os.remove(file_location)

        body["modelErrors"] = model_errors
        body["filename"] = file.filename
        return body

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
