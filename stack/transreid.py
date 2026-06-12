"""TransReID (ViT-base, SIE+JPM, MSMT17) embedding — vendored inference for the local checkpoint
`vit_transreid_msmt.pth`.

We use the GLOBAL branch BNNeck feature (768-d), the primary TransReID retrieval feature:
  base ViT (overlapping patch-embed stride-12, +SIE) -> b1 (one extra block + LayerNorm) -> cls token
  -> bottleneck (BNNeck BatchNorm1d) -> L2-normalize.
timm's `Block` matches the checkpoint's transformer keys (norm1/attn.qkv/attn.proj/norm2/mlp.fc1/mlp.fc2),
so we only hand-build the patch-embed, tokens, SIE, and the b1/bottleneck head, then load_state_dict(strict=False).
The JPM local branches (b2, bottleneck_1..4) and classifiers are intentionally unused for inference.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

CKPT = "vit_transreid_msmt.pth"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_LOCK = threading.Lock()
_MODEL = None


class _PatchEmbed(nn.Module):
    def __init__(self, dim, patch=16, stride=12):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=stride)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # B, N, dim


class _Base(nn.Module):
    def __init__(self, img=(256, 128), patch=16, stride=12, dim=768, depth=12, heads=12, cams=15):
        super().__init__()
        self.patch_embed = _PatchEmbed(dim, patch, stride)
        h = (img[0] - patch) // stride + 1
        w = (img[1] - patch) // stride + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, h * w + 1, dim))
        self.sie_embed = nn.Parameter(torch.zeros(cams, 1, dim))
        self.sie_xishu = 3.0
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio=4, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cam=0, use_sie=True):
        # JPM ("local_feature") mode: base runs blocks[:-1] and does NOT apply self.norm.
        # The 12th block lives in the head (b1/b2, trained copies of base.blocks[-1]); base.norm
        # and base.blocks[-1] are present in the checkpoint but unused on the inference path.
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed + (self.sie_xishu * self.sie_embed[cam] if use_sie else 0.0)
        for blk in self.blocks[:-1]:
            x = blk(x)
        return x


class TransReID(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.base = _Base(dim=dim, heads=heads)
        self.b1 = nn.Sequential(Block(dim, heads, mlp_ratio=4, qkv_bias=True), nn.LayerNorm(dim))
        self.bottleneck = nn.BatchNorm1d(dim)

    def forward(self, x, cam=0, use_sie=True):
        feat = self.base(x, cam, use_sie)
        g = self.b1(feat)[:, 0]            # global branch cls token
        return self.bottleneck(g)          # BNNeck retrieval feature


def _load() -> TransReID:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                m = TransReID().to(_DEV).eval()
                sd = torch.load(CKPT, map_location=_DEV, weights_only=False)
                miss, unexp = m.load_state_dict(sd, strict=False)
                real_miss = [k for k in miss if "num_batches_tracked" not in k]
                if real_miss:
                    print(f"[transreid] WARN missing weights: {real_miss[:6]}")
                _MODEL = m
    return _MODEL


def _prep(crop_bgr):
    im = cv2.cvtColor(cv2.resize(crop_bgr, (128, 256)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    im = (im - _MEAN) / _STD
    return torch.tensor(im.transpose(2, 0, 1)).unsqueeze(0).float().to(_DEV)


def transreid_embed(crop_bgr, cam: int = 0, use_sie: bool = False):
    """768-d L2-normalized TransReID global feature for one BGR crop (SIE off = better off-the-shelf)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    return embed_with(_load(), crop_bgr, use_sie, cam)


def load_finetuned(path):
    """Build TransReID, load the MSMT base, then overlay a fine-tuned delta checkpoint
    (from training.train_transreid). Returns (model, use_sie)."""
    ck = torch.load(path, map_location=_DEV, weights_only=False)
    m = TransReID().to(_DEV).eval()
    m.load_state_dict(torch.load(ck.get("base_ckpt", CKPT), map_location=_DEV, weights_only=False), strict=False)
    m.load_state_dict(ck["delta"], strict=False)
    return m, bool(ck.get("use_sie", False))


def embed_with(model, crop_bgr, use_sie: bool = False, cam: int = 0):
    """768-d L2-normalized feature for one BGR crop using an explicit (possibly fine-tuned) model."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    with torch.no_grad():
        f = model(_prep(crop_bgr), cam=cam, use_sie=use_sie).cpu().numpy().ravel()
    n = float(np.linalg.norm(f))
    return (f / n if n > 0 else f).astype("float32")
