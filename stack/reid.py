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


def _load_osnet(weights="osnet_x1_0_msmt17.pt"):
    global _osnet
    if _osnet is None:
        import torch
        from boxmot.reid.core.reid import ReID
        dev = "0" if torch.cuda.is_available() else "cpu"  # boxmot wants "0", not "cuda"
        _osnet = ReID(weights=weights, device=dev, half=False)
    return _osnet


def osnet_embed(crop):
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    reid = _load_osnet()
    h, w = crop.shape[:2]
    feats = reid(crop, boxes=np.array([[0, 0, w, h]], dtype=float))
    v = np.asarray(feats, dtype="float32").ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def osnet_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))
