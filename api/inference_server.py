"""
api/inference_server.py
========================
Production inference server with High-Fidelity Dynamic Resolution support.
"""

import io
import os
import time
import logging
from contextlib import asynccontextmanager

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Configuration from environment variables ─────────────────
HF_MODEL_REPO    = os.environ.get("HF_MODEL_REPO",    "")
HF_MODEL_FILE    = os.environ.get("HF_MODEL_FILE",    "best.pt")
LOCAL_CACHE_PATH = os.environ.get("MODEL_CACHE_PATH", "model_cache/best.pt")

logger = logging.getLogger("rrin-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Model holder ──────────────────────────────────────────────
class ModelHolder:
    generator = None
    device    = None
    is_ready  = False
    error_msg = None

_model = ModelHolder()


# ── Model loading ─────────────────────────────────────────────
def _download_model_from_huggingface() -> str:
    if os.path.exists(LOCAL_CACHE_PATH):
        return LOCAL_CACHE_PATH
    if not HF_MODEL_REPO:
        raise RuntimeError("HF_MODEL_REPO environment variable is not set.")
    
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(LOCAL_CACHE_PATH), exist_ok=True)
    return hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE,
        local_dir=os.path.dirname(LOCAL_CACHE_PATH),
        local_dir_use_symlinks=False,
    )

def _load_generator_inline(checkpoint_path: str, device: torch.device):
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvINAct(nn.Module):
        def __init__(self, ic, oc, k=4, s=2, p=1, act="lrelu", norm=True, tr=False):
            super().__init__()
            C = nn.ConvTranspose2d if tr else nn.Conv2d
            self.c = C(ic, oc, k, s, p, bias=not norm)
            self.n = nn.InstanceNorm2d(oc, affine=True) if norm else nn.Identity()
            self.a = (nn.LeakyReLU(0.2, True) if act == "lrelu" else
                      nn.ReLU(True) if act == "relu" else
                      nn.Tanh() if act == "tanh" else nn.Identity())
        def forward(self, x): return self.a(self.n(self.c(x)))

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.a = ConvINAct(ch, ch, 3, 1, 1, "relu")
            self.b = ConvINAct(ch, ch, 3, 1, 1, "none")
            self.d = nn.Dropout2d(0.15)
        def forward(self, x): return x + self.b(self.d(self.a(x)))

    class AttGate(nn.Module):
        def __init__(self, sc, gc, ic):
            super().__init__()
            self.wx = nn.Conv2d(sc, ic, 1)
            self.wg = nn.Conv2d(gc, ic, 1)
            self.ps = nn.Conv2d(ic, 1, 1)
        def forward(self, s, g):
            if g.shape[-2:] != s.shape[-2:]:
                g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
            return s * torch.sigmoid(self.ps(F.relu(self.wx(s) + self.wg(g))))

    class UNet(nn.Module):
        def __init__(self, ic=4, oc=3, F=64, N=6):
            super().__init__()
            self.e1 = ConvINAct(ic, F,   norm=False)
            self.e2 = ConvINAct(F,   F*2)
            self.e3 = ConvINAct(F*2, F*4)
            self.e4 = ConvINAct(F*4, F*8)
            self.e5 = ConvINAct(F*8, F*8)
            self.bn = nn.Sequential(*[ResBlock(F*8) for _ in range(N)])
            self.a4 = AttGate(F*8, F*8, F*4)
            self.a3 = AttGate(F*4, F*4, F*2)
            self.a2 = AttGate(F*2, F*2, F)
            self.a1 = AttGate(F,   F,   F//2)
            self.d1 = ConvINAct(F*8,  F*8, act="relu", tr=True)
            self.d2 = ConvINAct(F*16, F*4, act="relu", tr=True)
            self.d3 = ConvINAct(F*8,  F*2, act="relu", tr=True)
            self.d4 = ConvINAct(F*4,  F,   act="relu", tr=True)
            self.out= ConvINAct(F*2,  oc,  act="tanh", norm=False, tr=True)
        def forward(self, x):
            e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3); e5=self.e5(e4)
            b=self.bn(e5)
            d1=self.d1(b)
            d2=self.d2(torch.cat([d1, self.a4(e4,d1)],1))
            d3=self.d3(torch.cat([d2, self.a3(e3,d2)],1))
            d4=self.d4(torch.cat([d3, self.a2(e2,d3)],1))
            return self.out(torch.cat([d4, self.a1(e1,d4)],1))

    payload = torch.load(checkpoint_path, map_location=device)
    generator = UNet().to(device)

    state_key = "generator_state" if "generator_state" in payload else None
    state     = payload[state_key] if state_key else payload

    translated_state = {}
    for key, value in state.items():
        k = key
        k = k.replace('encoder_', 'e')
        k = k.replace('decoder_', 'd')
        k = k.replace('bottleneck.', 'bn.')
        k = k.replace('attn_', 'a')
        k = k.replace('.conv_a.', '.a.')
        k = k.replace('.conv_b.', '.b.')
        k = k.replace('.conv.', '.c.')
        k = k.replace('.norm.', '.n.')
        k = k.replace('.w_x.', '.wx.')
        k = k.replace('.w_g.', '.wg.')
        k = k.replace('.psi.', '.ps.')
        k = k.replace('output_layer.', 'out.')
        translated_state[k] = value

    generator.load_state_dict(translated_state, strict=False)
    generator.eval()
    return generator, device


# ── HIGH-FIDELITY IMAGE PROCESSING ───────────────────────────
def pil_image_to_4ch_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Dynamically scales image to a clean multiple of 32 to eliminate blur."""
    img_rgb = img.convert("RGB")
    orig_w, orig_h = img_rgb.size
    
    # Target maximum boundary of 1024px to protect CPU memory limits while preserving high resolution
    max_dim = 1024
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        nw = int(round((orig_w * scale) / 32.0)) * 32
        nh = int(round((orig_h * scale) / 32.0)) * 32
    else:
        nw = int(round(orig_w / 32.0)) * 32
        nh = int(round(orig_h / 32.0)) * 32
        
    nw = max(nw, 32)
    nh = max(nh, 32)
    
    # High-quality resize to full processing resolution
    img_resized = img_rgb.resize((nw, nh), Image.LANCZOS)
    
    # Convert and normalize
    t = T.ToTensor()(img_resized)
    t = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(t)
    
    # Extract and cat green channel
    green = t[1:2, :, :]
    t4    = torch.cat([t, green], dim=0).unsqueeze(0)
    
    return t4.to(device)


def tensor_to_pil(tensor: torch.Tensor, original_size: tuple) -> Image.Image:
    """Converts high-resolution tensor back to standard PIL image space cleanly."""
    t = tensor.squeeze(0).detach().cpu()
    t = (t + 1.0) / 2.0  # Undo Tanh scale
    
    img = T.ToPILImage()(t.clamp(0.0, 1.0))
    
    # Restore perfectly to original uploaded dimensions
    if img.size != original_size:
        img = img.resize(original_size, Image.LANCZOS)
    return img


# ── FastAPI Application ───────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        checkpoint_path      = _download_model_from_huggingface()
        gen, dev             = _load_generator_inline(checkpoint_path, torch.device("cpu"))
        _model.generator     = gen
        _model.device        = dev
        _model.is_ready      = True
        logger.info("Model loaded and ready.")
    except Exception as e:
        _model.error_msg = str(e)
        logger.error(f"FAILED to load model: {e}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/health")
async def health():
    return {"status": "ready" if _model.is_ready else "unavailable"}

@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    if not _model.is_ready:
        raise HTTPException(status_code=503, detail="Model loading")

    try:
        content = await file.read()
        img     = Image.open(io.BytesIO(content))
        orig_size = img.size

        start = time.time()
        with torch.no_grad():
            inp = pil_image_to_4ch_tensor(img, _model.device)
            out = _model.generator(inp)
        elapsed_ms = (time.time() - start) * 1000

        restored_img = tensor_to_pil(out, orig_size)
        buf = io.BytesIO()
        restored_img.save(buf, format="PNG", optimize=False)
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="image/png",
            headers={"X-Processing-Time-Ms": f"{elapsed_ms:.0f}"},
        )
    except Exception as e:
        logger.error(f"Restoration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.inference_server:app", host="0.0.0.0", port=7860, reload=False)