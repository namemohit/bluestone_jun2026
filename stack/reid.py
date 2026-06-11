"""Appearance embedder for ReID-style matching (v2).

torchvision ResNet18 penultimate features (512-d, L2-normalized). Far more discriminative
than a colour histogram -- it keys on clothing texture + body shape, so it won't confuse a
woman for a man. A dedicated person-ReID model (OSNet / CLIP-ReID via torchreid/boxmot) is
the quality upgrade and is view-invariant (better front<->back); this avoids extra installs
and is already a big step up from colour-hist for disambiguation.
"""
from __future__ import annotations

import cv2
import numpy as np
# torch / torchvision / boxmot are imported LAZILY (only when an actual embedding is computed) so that
# a cached L4 re-run — which only calls the numpy osnet_sim — doesn't pay the ~6s torch import.

_model = _tf = _dev = None


def _load():
    global _model, _tf, _dev
    if _model is None:
        import torch
        import torchvision.models as M
        import torchvision.transforms as T
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()  # -> 512-d feature vector
        _model = m.eval().to(_dev)
        _tf = T.Compose([
            T.ToTensor(),
            T.Resize((256, 128), antialias=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _model, _tf, _dev


def embed(crop):
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    import torch
    m, tf, dev = _load()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = tf(rgb).unsqueeze(0).to(dev)
    with torch.no_grad():
        f = m(x).cpu().numpy().ravel()
    return (f / (np.linalg.norm(f) + 1e-9)).astype("float32")


def sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))  # cosine (both L2-normalized)


# --- OSNet person-ReID (boxmot): person-focused + view-invariant, fine-tunable on site data ---
_osnet = None

# A promoted ReID model is signalled by a sentinel file so EVERY embedder in every process
# (dashboard, re-embed, future-day stack) uses the same model -> cache + gallery + live processing
# share ONE embedding space. Remove the sentinel to fall back to stock OSNet. Two sentinel forms:
#   boxmot:<weights_name>   e.g. boxmot:osnet_ain_x1_0_msmt17.pt  (boxmot factory, auto-downloads)
#   <abs path to .pt>       a local contrastive fine-tune checkpoint (training/finetune_triplet.py)
import os
from pathlib import Path

_ACTIVE_WEIGHTS_FILE = Path(__file__).resolve().parents[1] / "outputs" / "reid_active_weights.txt"
_triplet = None        # (model, dev) for a promoted contrastive osnet, cached
_triplet_path = None   # which weights file the cached _triplet was built from


def _active_weights():
    """The promoted model from the sentinel: ('boxmot', name) | ('triplet', path) | None (stock)."""
    try:
        if _ACTIVE_WEIGHTS_FILE.exists():
            p = _ACTIVE_WEIGHTS_FILE.read_text(encoding="utf-8").strip()
            if p.startswith("boxmot:"):
                return ("boxmot", p[len("boxmot:"):].strip())
            if p and Path(p).exists():
                return ("triplet", p)
    except Exception:
        pass
    return None


def _active_triplet_weights():
    """Back-compat shim: the promoted contrastive checkpoint path, or None."""
    aw = _active_weights()
    return aw[1] if aw and aw[0] == "triplet" else None


def _load_triplet(weights):
    global _triplet, _triplet_path
    if _triplet is None or _triplet_path != weights:
        import torch
        from boxmot.reid.backbones.osnet import osnet_x1_0
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = osnet_x1_0(num_classes=0, pretrained=False, loss="triplet")
        ckpt = torch.load(weights, map_location=dev, weights_only=False)  # our own trusted checkpoint
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        m.load_state_dict(sd, strict=False)
        _triplet = (m.eval().to(dev), dev)
        _triplet_path = weights
    return _triplet


def _triplet_embed(crop, weights):
    """Embed with the promoted contrastive model. Preprocessing MUST match
    training/finetune_triplet.py:_load_image so the cache stays in one space."""
    import torch
    m, dev = _load_triplet(weights)
    im = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = torch.tensor(im.astype("float32")).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        o = m(x)
        o = o[-1] if isinstance(o, (tuple, list)) else o   # triplet head returns a tensor in eval
        f = (o / (o.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy().ravel()
    return f.astype("float32")


def reset_osnet():
    """Drop cached models so the next embed reloads with the currently-active weights."""
    global _osnet, _osnet_weights, _triplet, _triplet_path
    _osnet = _osnet_weights = _triplet = _triplet_path = None


_osnet_weights = None  # which boxmot weights the cached _osnet was built with


def _load_osnet(weights="osnet_x1_0_msmt17.pt"):
    global _osnet, _osnet_weights
    if _osnet is None or _osnet_weights != weights:
        import torch
        from boxmot.reid.core.reid import ReID
        dev = "0" if torch.cuda.is_available() else "cpu"  # boxmot wants "0", not "cuda"
        _osnet = ReID(weights=Path(weights), device=dev, half=False)
        _osnet_weights = weights
    return _osnet


def osnet_embed(crop):
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    aw = _active_weights()
    if aw and aw[0] == "triplet":          # promoted local contrastive checkpoint
        return _triplet_embed(crop, aw[1])
    reid = _load_osnet(aw[1] if aw else "osnet_x1_0_msmt17.pt")   # promoted boxmot weights, else stock
    h, w2 = crop.shape[:2]
    feats = reid(crop, boxes=np.array([[0, 0, w2, h]], dtype=float))
    v = np.asarray(feats, dtype="float32").ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def osnet_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))
