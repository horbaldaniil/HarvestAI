"""Environment check for the thesis segmentation work.

Prints library versions and the selected torch device, then exercises it:
  * CUDA  → a few training steps of a tiny encoder-decoder in bf16 autocast
            (the Windows / RTX 5070 Ti training box);
  * MPS / CPU → a forward pass only (the Apple Silicon demo Mac),
            or training too with --train.

Exit code is non-zero if any check fails, so it can gate a demo setup.

Run (from backend/):
    uv run python scripts/thesis/check_env.py
    uv run python scripts/thesis/check_env.py --train   # force a training check
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from app.segmentation.device import get_device

GEO_LIBS = ("rasterio", "geopandas", "shapely", "skimage", "tensorboard", "numpy")

IN_BANDS = 8
TILE = 256
BATCH = 8


class TinyUNet(nn.Module):
    """Two-level encoder-decoder — just enough to exercise conv, pooling,
    upsampling and skip connections on the device."""

    def __init__(self, c_in: int = IN_BANDS, c: int = 32, n_out: int = 3):
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )

        self.enc1 = block(c_in, c)
        self.enc2 = block(c, c * 2)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = block(c * 2, c)
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(torch.cat([self.up(e2), e1], dim=1))
        return self.head(d1)


def check_imports() -> bool:
    ok = True
    for name in GEO_LIBS:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:<12} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 — report every failure, keep going
            print(f"  {name:<12} FAILED: {exc}")
            ok = False
    try:
        import rasterio

        print(f"  {'GDAL':<12} {rasterio.__gdal_version__}")
    except Exception:  # noqa: BLE001
        pass
    return ok


def check_tensorboard() -> bool:
    from torch.utils.tensorboard import SummaryWriter

    with tempfile.TemporaryDirectory() as tmp:
        writer = SummaryWriter(log_dir=tmp)
        writer.add_scalar("check/loss", 1.0, 0)
        writer.close()
        written = any(Path(tmp).rglob("events.out.tfevents.*"))
    print(f"  tensorboard writer: {'ok' if written else 'FAILED'}")
    return written


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def check_inference(device: torch.device) -> bool:
    model = TinyUNet().to(device).eval()
    x = torch.randn(BATCH, IN_BANDS, TILE, TILE, device=device)
    with torch.inference_mode():
        model(x)  # warm-up
        sync(device)
        t0 = time.perf_counter()
        y = model(x)
        sync(device)
    dt = time.perf_counter() - t0
    ok = tuple(y.shape) == (BATCH, 3, TILE, TILE) and bool(torch.isfinite(y).all())
    print(f"  inference: {BATCH}×{IN_BANDS}×{TILE}×{TILE} in {dt * 1000:.0f} ms — "
          f"{'ok' if ok else 'FAILED'}")
    return ok


def check_training(device: torch.device, steps: int = 30) -> bool:
    torch.manual_seed(0)
    model = TinyUNet().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(BATCH, IN_BANDS, TILE, TILE, device=device)
    target = (x[:, :3] > 0).float()  # learnable toy target
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses = []
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    sync(device)
    dt = time.perf_counter() - t0

    ok = losses[-1] < losses[0] and all(v == v for v in losses)
    mem = ""
    if device.type == "cuda":
        mem = f", peak mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    print(f"  training: {steps} steps{' bf16' if use_amp else ''} in {dt:.2f} s "
          f"({steps / dt:.1f} it/s{mem}), loss {losses[0]:.3f} → {losses[-1]:.3f} — "
          f"{'ok' if ok else 'FAILED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true", help="run the training check on MPS/CPU too")
    args = ap.parse_args()

    print(f"OS:     {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch:  {torch.__version__} (CUDA build: {torch.version.cuda or 'none'})")

    device = get_device()
    print(f"device: {device}")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU {props.name}, {props.total_memory / 2**30:.1f} GiB, sm_{cap[0]}{cap[1]}")
        if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
            print("  WARNING: this torch build has no kernels for the GPU's architecture")

    print("libraries:")
    results = [check_imports(), check_tensorboard()]
    print("compute:")
    results.append(check_inference(device))
    if device.type == "cuda" or args.train:
        results.append(check_training(device))

    ok = all(results)
    print("RESULT:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
