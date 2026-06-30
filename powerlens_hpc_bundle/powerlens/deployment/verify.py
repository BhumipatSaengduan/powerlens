"""
ONNX Numerical Equivalence Verification
========================================
ตรวจว่า output ของ ONNX models = output ของ PyTorch models (within tolerance)

ทำไมต้อง verify:
    - Export อาจ silent fail บางส่วน (custom ops, dropout layers, etc.)
    - Numerical precision ระหว่าง PyTorch และ ONNX อาจต่างกันเล็กน้อย
    - Dynamic shapes (batch size = 1, 4, 32) ต้องทำงานทุก batch size
    - Random inputs + edge cases (zeros, large values) ต้องผ่าน

What we check:
    1. Output shape ตรงกัน
    2. Output values ใกล้เคียง (atol=1e-4, rtol=1e-3)
    3. ทำงานได้กับ batch sizes ต่างๆ (1, 4, 32)
    4. ไม่มี NaN/Inf ใน outputs

Usage:
    python -m powerlens.deployment.verify \\
        --checkpoint runs/exp1/checkpoints/best.pt \\
        --onnx-dir models/drl_stfn_v1/

Returns exit code 0 ถ้า pass, 1 ถ้า fail
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import onnxruntime as ort

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from powerlens.models.trainer import DRLSTFNTrainer
from powerlens.models.config import ROUTER_CONFIG, EXPERT_CONFIG, RouterConfig, ExpertConfig


# Tolerance for numerical equivalence
ATOL_DEFAULT = 1e-4    # absolute tolerance
RTOL_DEFAULT = 1e-3    # relative tolerance


# ============================================================
# Verification helpers
# ============================================================

def _compare_tensors(
    pytorch_out: np.ndarray,
    onnx_out: np.ndarray,
    name: str,
    atol: float,
    rtol: float,
) -> Tuple[bool, Dict]:
    """
    Compare 2 numpy arrays — return (passed, diagnostic_info).
    """
    info = {
        "name": name,
        "pytorch_shape": list(pytorch_out.shape),
        "onnx_shape": list(onnx_out.shape),
    }

    # Shape check
    if pytorch_out.shape != onnx_out.shape:
        info["error"] = f"Shape mismatch: PyTorch {pytorch_out.shape} vs ONNX {onnx_out.shape}"
        return False, info

    # NaN/Inf check
    if np.isnan(pytorch_out).any() or np.isnan(onnx_out).any():
        info["error"] = "NaN detected"
        return False, info
    if np.isinf(pytorch_out).any() or np.isinf(onnx_out).any():
        info["error"] = "Inf detected"
        return False, info

    # Numerical comparison
    abs_diff = np.abs(pytorch_out - onnx_out)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())

    # Relative diff (safe div)
    denom = np.maximum(np.abs(pytorch_out), 1e-9)
    rel_diff = abs_diff / denom
    max_rel_diff = float(rel_diff.max())

    info["max_abs_diff"] = max_abs_diff
    info["mean_abs_diff"] = mean_abs_diff
    info["max_rel_diff"] = max_rel_diff

    # Pass if within tolerance
    passed = bool(np.allclose(pytorch_out, onnx_out, atol=atol, rtol=rtol))
    if not passed:
        info["error"] = f"Numerical mismatch: max_abs_diff={max_abs_diff:.2e}, " \
                        f"max_rel_diff={max_rel_diff:.2e} (atol={atol}, rtol={rtol})"
    return passed, info


# ============================================================
# Per-model verification
# ============================================================

def verify_router(
    pytorch_router,
    onnx_path: Path,
    seq_len: int,
    n_features: int,
    batch_sizes: List[int],
    atol: float = ATOL_DEFAULT,
    rtol: float = RTOL_DEFAULT,
) -> Tuple[bool, List[Dict]]:
    """
    Verify Router ONNX output matches PyTorch.
    
    Tests:
        - Multiple batch sizes
        - Random inputs (different seeds)
    """
    pytorch_router.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    results = []
    overall_passed = True

    for bs in batch_sizes:
        # Use deterministic input
        torch.manual_seed(42 + bs)
        x_pt = torch.randn(bs, seq_len, n_features)
        x_np = x_pt.numpy().astype(np.float32)

        # PyTorch forward
        with torch.no_grad():
            pt_out = pytorch_router(x_pt).numpy()

        # ONNX forward
        ort_out = session.run(None, {"features": x_np})[0]

        passed, info = _compare_tensors(
            pt_out, ort_out, f"router_bs{bs}", atol, rtol,
        )
        info["batch_size"] = bs
        results.append(info)
        overall_passed = overall_passed and passed

        marker = "✓" if passed else "✗"
        if passed:
            print(f"    {marker} bs={bs:3d}: max_abs={info['max_abs_diff']:.2e}, "
                  f"max_rel={info['max_rel_diff']:.2e}")
        else:
            print(f"    {marker} bs={bs:3d}: {info.get('error', 'FAIL')}")

    return overall_passed, results


def verify_expert(
    pytorch_expert,
    onnx_path: Path,
    category: str,
    seq_len: int,
    n_features: int,
    batch_sizes: List[int],
    atol: float = ATOL_DEFAULT,
    rtol: float = RTOL_DEFAULT,
) -> Tuple[bool, List[Dict]]:
    """
    Verify Expert ONNX output matches PyTorch.
    
    PyTorch expert returns 3 tensors (status, power, current).
    ONNX returns single concatenated (B, 3).
    """
    pytorch_expert.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    results = []
    overall_passed = True

    for bs in batch_sizes:
        torch.manual_seed(42 + bs)
        x_pt = torch.randn(bs, seq_len, n_features)
        x_np = x_pt.numpy().astype(np.float32)

        # PyTorch forward → 3 tensors → concat dim=1
        with torch.no_grad():
            s, p, c = pytorch_expert(x_pt)
            pt_out = torch.cat([s, p, c], dim=1).numpy()

        # ONNX forward
        ort_out = session.run(None, {"features": x_np})[0]

        passed, info = _compare_tensors(
            pt_out, ort_out, f"expert_{category}_bs{bs}", atol, rtol,
        )
        info["batch_size"] = bs
        results.append(info)
        overall_passed = overall_passed and passed

        marker = "✓" if passed else "✗"
        if passed:
            print(f"    {marker} bs={bs:3d}: max_abs={info['max_abs_diff']:.2e}, "
                  f"max_rel={info['max_rel_diff']:.2e}")
        else:
            print(f"    {marker} bs={bs:3d}: {info.get('error', 'FAIL')}")

    return overall_passed, results


# ============================================================
# Main verification pipeline
# ============================================================

def verify_all(
    checkpoint_path: str,
    onnx_dir: str,
    batch_sizes: List[int] = None,
    atol: float = ATOL_DEFAULT,
    rtol: float = RTOL_DEFAULT,
) -> Tuple[bool, Dict]:
    """
    Full verification: load PyTorch + ONNX, compare across all models + batch sizes.
    
    Returns:
        (overall_passed, full_report dict)
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 32]

    onnx_dir = Path(onnx_dir)
    manifest_path = onnx_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {onnx_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nLoading PyTorch checkpoint: {checkpoint_path}")
    ckpt_meta = torch.load(checkpoint_path, map_location="cpu")
    router_cfg = RouterConfig(**ckpt_meta.get("router_config", {
        "seq_len": ROUTER_CONFIG.seq_len,
        "n_features": ROUTER_CONFIG.n_features,
        "categories": ROUTER_CONFIG.categories,
        "sftn_channels": ROUTER_CONFIG.sftn_channels,
        "sftn_kernel": ROUTER_CONFIG.sftn_kernel,
        "sftn_blocks": ROUTER_CONFIG.sftn_blocks,
        "gru_hidden": ROUTER_CONFIG.gru_hidden,
        "gru_layers": ROUTER_CONFIG.gru_layers,
        "bidirectional": ROUTER_CONFIG.bidirectional,
        "trunk_hidden": ROUTER_CONFIG.trunk_hidden,
        "head_hidden": ROUTER_CONFIG.head_hidden,
        "n_actions_per_head": ROUTER_CONFIG.n_actions_per_head,
    }))
    expert_cfg = ExpertConfig(**ckpt_meta.get("expert_config", {
        "seq_len": EXPERT_CONFIG.seq_len,
        "n_features": EXPERT_CONFIG.n_features,
        "sftn_channels": EXPERT_CONFIG.sftn_channels,
        "sftn_kernel": EXPERT_CONFIG.sftn_kernel,
        "gru_hidden": EXPERT_CONFIG.gru_hidden,
        "gru_layers": EXPERT_CONFIG.gru_layers,
        "gru_dropout": EXPERT_CONFIG.gru_dropout,
        "attn_heads": EXPERT_CONFIG.attn_heads,
        "attn_dropout": EXPERT_CONFIG.attn_dropout,
        "head_hidden": EXPERT_CONFIG.head_hidden,
        "categories": EXPERT_CONFIG.categories,
    }))
    trainer = DRLSTFNTrainer(
        router_config=router_cfg,
        expert_config=expert_cfg,
        device="cpu",
    )
    trainer.load_checkpoint(checkpoint_path)

    full_report = {
        "checkpoint": checkpoint_path,
        "onnx_dir": str(onnx_dir),
        "batch_sizes_tested": batch_sizes,
        "atol": atol,
        "rtol": rtol,
        "models": {},
    }
    overall_passed = True

    # Verify Router
    print(f"\n[Router] {onnx_dir / 'router.onnx'}")
    passed, results = verify_router(
        trainer.router, onnx_dir / "router.onnx",
        seq_len=router_cfg.seq_len,
        n_features=router_cfg.n_features,
        batch_sizes=batch_sizes, atol=atol, rtol=rtol,
    )
    full_report["models"]["router"] = {"passed": passed, "results": results}
    overall_passed = overall_passed and passed

    # Verify Experts
    for cat in expert_cfg.categories:
        cat_filename = cat.lower().replace(" ", "_")
        onnx_path = onnx_dir / f"expert_{cat_filename}.onnx"
        print(f"\n[Expert {cat}] {onnx_path}")
        passed, results = verify_expert(
            trainer.experts[cat], onnx_path, category=cat,
            seq_len=expert_cfg.seq_len,
            n_features=expert_cfg.n_features,
            batch_sizes=batch_sizes, atol=atol, rtol=rtol,
        )
        full_report["models"][f"expert_{cat}"] = {"passed": passed, "results": results}
        overall_passed = overall_passed and passed

    full_report["overall_passed"] = overall_passed

    # Summary
    print(f"\n{'='*60}")
    print(f"Verification {'PASSED ✓' if overall_passed else 'FAILED ✗'}")
    print(f"{'='*60}")
    print(f"Models tested: {len(full_report['models'])}")
    print(f"Batch sizes:   {batch_sizes}")
    print(f"Tolerance:     atol={atol}, rtol={rtol}")

    n_passed = sum(1 for m in full_report["models"].values() if m["passed"])
    print(f"Passed:        {n_passed}/{len(full_report['models'])}")

    return overall_passed, full_report


def main():
    parser = argparse.ArgumentParser(description="DRL-STFN ONNX Numerical Verification")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--onnx-dir", type=str, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 32])
    parser.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    parser.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    parser.add_argument("--save-report", type=str, default=None,
                        help="Optional path to save JSON report")
    args = parser.parse_args()

    passed, report = verify_all(
        args.checkpoint, args.onnx_dir,
        batch_sizes=args.batch_sizes,
        atol=args.atol, rtol=args.rtol,
    )

    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved: {args.save_report}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
