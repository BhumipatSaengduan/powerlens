"""
PyTorch → ONNX Export
======================
Convert trained DRL-STFN checkpoint → 5 ONNX files:
    - router.onnx
    - expert_plug.onnx
    - expert_light.onnx
    - expert_ac.onnx
    - expert_water_heater.onnx

ทำไมแยก 5 ไฟล์:
    - Hot-swap ได้ทีละตัว (ภายหลังถ้า AC retrain เสร็จก่อน upgrade เฉพาะ AC)
    - Inference orchestration อยู่ที่ Python wrapper (ไม่ต้อง bake routing เข้า graph)
    - Online adaptive learning ของแต่ละ expert ไม่กระทบกัน

ทำไม Confidence/Constraint/Fusion ไม่ export:
    - Logic เป็น control flow (if/else, retry loop) — ONNX ทำได้ยาก
    - เปลี่ยน threshold ได้โดยไม่ต้อง re-export
    - Stay in Python wrapper เป็นเรื่องที่เหมาะสม

Usage:
    python -m powerlens.deployment.export \\
        --checkpoint runs/exp1/checkpoints/best.pt \\
        --output-dir models/drl_stfn_v1/ \\
        --opset 17

Output:
    models/drl_stfn_v1/
    ├── router.onnx
    ├── expert_plug.onnx
    ├── expert_light.onnx
    ├── expert_ac.onnx
    ├── expert_water_heater.onnx
    └── manifest.json     ← metadata: version, shapes, categories, hash
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

# Robust path setup for both module and direct invocation
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from powerlens.models.trainer import DRLSTFNTrainer
from powerlens.models.config import (
    ROUTER_CONFIG, EXPERT_CONFIG, RouterConfig, ExpertConfig,
)


# ============================================================
# Model wrappers — clean inputs/outputs for ONNX
# ============================================================

class RouterONNXWrapper(nn.Module):
    """
    Wraps BranchingDQNRouter for ONNX export.
    
    Output: Q-values only (action selection done in Python wrapper).
    เหตุผล: argmax → action mask logic ทำใน Python ดีกว่า — เปลี่ยน
    epsilon strategy ได้โดยไม่ต้อง re-export
    
    Input:  features (B, T, F)
    Output: q_values (B, N_categories, 2)
    """
    def __init__(self, router):
        super().__init__()
        self.router = router

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.router(features)


class ExpertONNXWrapper(nn.Module):
    """
    Wraps single DRLSTFNExpert for ONNX export.
    
    Stack 3 outputs (status, power, current) → single tensor (B, 3)
    เพื่อ ONNX schema ง่ายขึ้น (1 output แทน 3)
    
    Input:  features (B, T, F)
    Output: predictions (B, 3) = [status, power, current]
    """
    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        status, power, current = self.expert(features)
        # Stack to (B, 3) — model returns (B, 1) each, concat dim=1
        return torch.cat([status, power, current], dim=1)


# ============================================================
# Export functions
# ============================================================

def _file_sha256(path: Path) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_router(
    router,
    output_path: Path,
    seq_len: int,
    n_features: int,
    opset_version: int = 17,
) -> Dict:
    """
    Export Router → router.onnx
    
    Returns metadata dict.
    """
    print(f"  Exporting Router → {output_path.name}")
    router.eval()
    wrapper = RouterONNXWrapper(router)
    wrapper.eval()

    dummy_input = torch.randn(1, seq_len, n_features)

    # Verify forward works ก่อน export
    with torch.no_grad():
        _ = wrapper(dummy_input)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["features"],
        output_names=["q_values"],
        dynamic_axes={
            "features": {0: "batch"},
            "q_values": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,    # use legacy exporter (stable, supports dynamic_axes well)
    )

    return {
        "file": output_path.name,
        "type": "router",
        "input_shape": [None, seq_len, n_features],
        "input_names": ["features"],
        "output_names": ["q_values"],
        "output_shape": [None, len(router.categories), 2],
        "categories": list(router.categories),
        "sha256": _file_sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def export_expert(
    expert,
    output_path: Path,
    category: str,
    seq_len: int,
    n_features: int,
    opset_version: int = 17,
) -> Dict:
    """Export single Expert → expert_<category>.onnx"""
    print(f"  Exporting Expert[{category}] → {output_path.name}")
    expert.eval()
    wrapper = ExpertONNXWrapper(expert)
    wrapper.eval()

    dummy_input = torch.randn(1, seq_len, n_features)

    with torch.no_grad():
        _ = wrapper(dummy_input)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["features"],
        output_names=["predictions"],
        dynamic_axes={
            "features": {0: "batch"},
            "predictions": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    return {
        "file": output_path.name,
        "type": "expert",
        "category": category,
        "input_shape": [None, seq_len, n_features],
        "input_names": ["features"],
        "output_names": ["predictions"],
        "output_shape": [None, 3],
        "output_columns": ["status", "power", "current"],
        "sha256": _file_sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def export_all(
    checkpoint_path: str,
    output_dir: str,
    opset_version: int = 17,
    device: str = "cpu",
) -> Dict:
    """
    Full export pipeline: load checkpoint → export 5 ONNX files → write manifest.
    
    Args:
        checkpoint_path: path to .pt checkpoint
        output_dir:      where to write .onnx + manifest.json
        opset_version:   ONNX opset (17 = stable, supports MultiheadAttention)
        device:          loading device (export ที่ CPU พอ — เร็วและ portable)
    Returns:
        manifest dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt_meta = torch.load(checkpoint_path, map_location=device)
    router_cfg = RouterConfig(**ckpt_meta.get("router_config", asdict(ROUTER_CONFIG)))
    expert_cfg = ExpertConfig(**ckpt_meta.get("expert_config", asdict(EXPERT_CONFIG)))
    feature_meta = ckpt_meta.get("feature_config") or {}
    feature_names = feature_meta.get("feature_names")
    if not feature_names:
        default_names = [
            "V_rms", "I_rms", "P", "Q", "PF", "THD",
            "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10",
        ]
        feature_names = (
            default_names
            if router_cfg.n_features == len(default_names)
            else [f"feature_{i}" for i in range(router_cfg.n_features)]
        )

    trainer = DRLSTFNTrainer(
        router_config=router_cfg,
        expert_config=expert_cfg,
        device=device,
    )
    trainer.load_checkpoint(checkpoint_path)
    trainer.router.eval()
    trainer.experts.eval()

    print(f"Output directory: {output_dir}")
    print(f"ONNX opset version: {opset_version}\n")

    artifacts = []

    # Export Router
    router_path = output_dir / "router.onnx"
    artifacts.append(export_router(
        trainer.router, router_path,
        seq_len=router_cfg.seq_len,
        n_features=router_cfg.n_features,
        opset_version=opset_version,
    ))

    # Export each Expert
    for cat in expert_cfg.categories:
        expert = trainer.experts[cat]
        # Filename: lowercase, replace _ with _ (already snake_case)
        cat_filename = cat.lower().replace(" ", "_")
        expert_path = output_dir / f"expert_{cat_filename}.onnx"
        artifacts.append(export_expert(
            expert, expert_path, category=cat,
            seq_len=expert_cfg.seq_len,
            n_features=expert_cfg.n_features,
            opset_version=opset_version,
        ))

    # Build manifest
    manifest = {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "opset_version": opset_version,
        "categories": list(router_cfg.categories),
        "input_spec": {
            "seq_len": router_cfg.seq_len,
            "n_features": router_cfg.n_features,
            "feature_names": feature_names,
        },
        "configs": {
            "router": asdict(router_cfg),
            "expert": asdict(expert_cfg),
        },
        "artifacts": artifacts,
        "total_size_bytes": sum(a["size_bytes"] for a in artifacts),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"Export Complete")
    print(f"{'='*60}")
    print(f"Files exported: {len(artifacts)}")
    for a in artifacts:
        size_kb = a["size_bytes"] / 1024
        print(f"  {a['file']:35s} {size_kb:7.1f} KB")
    total_mb = manifest["total_size_bytes"] / 1024 / 1024
    print(f"  {'TOTAL':35s} {total_mb:7.2f} MB")
    print(f"\nManifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="DRL-STFN PyTorch → ONNX Export")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for .onnx files")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default 17)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for loading (cpu recommended)")
    args = parser.parse_args()

    export_all(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        opset_version=args.opset,
        device=args.device,
    )


if __name__ == "__main__":
    main()
