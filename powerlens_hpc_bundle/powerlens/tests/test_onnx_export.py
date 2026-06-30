"""
Tests for ONNX Export Pipeline.

Coverage:
  Export:
    1. export_router() creates valid ONNX file
    2. export_expert() creates valid ONNX file
    3. export_all() creates 5 ONNX + manifest.json
    4. Manifest schema valid
    5. ONNX file size reasonable (< 50 MB total)
  
  Verification:
    6. verify_router() passes for fresh export
    7. verify_expert() passes for fresh export
    8. verify_all() returns True for clean export
    9. Multiple batch sizes work (dynamic axes)
    10. ONNX files load successfully with onnxruntime
"""
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import onnx
import onnxruntime as ort

from powerlens.deployment.export import (
    export_router, export_expert, export_all,
    RouterONNXWrapper, ExpertONNXWrapper,
)
from powerlens.deployment.verify import (
    verify_router, verify_expert, verify_all,
)
from powerlens.models.trainer import DRLSTFNTrainer
from powerlens.models.config import ROUTER_CONFIG, EXPERT_CONFIG


# ============================================================
# Helper: build & save a fresh trainer to checkpoint
# ============================================================

def _build_checkpoint(tmp_dir: Path) -> Path:
    """Create a freshly-initialized trainer + save checkpoint."""
    trainer = DRLSTFNTrainer(device="cpu")
    ckpt_path = tmp_dir / "test_ckpt.pt"
    trainer.save_checkpoint(str(ckpt_path))
    return ckpt_path


# ============================================================
# Export tests
# ============================================================

def test_router_export_creates_valid_onnx():
    """export_router() produces ONNX file ที่ load ได้ + has expected I/O"""
    print("\n=== Test 1: Router Export ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trainer = DRLSTFNTrainer(device="cpu")

        out_path = tmp / "router.onnx"
        meta = export_router(
            trainer.router, out_path,
            seq_len=ROUTER_CONFIG.seq_len,
            n_features=ROUTER_CONFIG.n_features,
        )

        # File exists
        assert out_path.exists()
        # ONNX validates
        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)

        # I/O names ตรง
        assert meta["input_names"] == ["features"]
        assert meta["output_names"] == ["q_values"]

        size_kb = meta["size_bytes"] / 1024
        print(f"  ✓ router.onnx valid, {size_kb:.1f} KB")
        print(f"  ✓ I/O: features → q_values")


def test_expert_export_creates_valid_onnx():
    """export_expert() produces valid ONNX with concatenated 3-output"""
    print("\n=== Test 2: Expert Export ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trainer = DRLSTFNTrainer(device="cpu")
        expert = trainer.experts["AC"]

        out_path = tmp / "expert_ac.onnx"
        meta = export_expert(
            expert, out_path, category="AC",
            seq_len=EXPERT_CONFIG.seq_len,
            n_features=EXPERT_CONFIG.n_features,
        )

        assert out_path.exists()
        model = onnx.load(str(out_path))
        onnx.checker.check_model(model)

        assert meta["category"] == "AC"
        assert meta["output_columns"] == ["status", "power", "current"]

        size_kb = meta["size_bytes"] / 1024
        print(f"  ✓ expert_ac.onnx valid, {size_kb:.1f} KB")
        print(f"  ✓ Output schema: {meta['output_columns']}")


def test_export_all_creates_5_files_plus_manifest():
    """export_all() creates exactly 5 .onnx + manifest.json"""
    print("\n=== Test 3: Export All ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = _build_checkpoint(tmp)
        out_dir = tmp / "models"

        manifest = export_all(
            checkpoint_path=str(ckpt),
            output_dir=str(out_dir),
            opset_version=17,
        )

        # 5 ONNX files
        onnx_files = list(out_dir.glob("*.onnx"))
        assert len(onnx_files) == 5, f"Expected 5 ONNX, got {len(onnx_files)}"
        expected = {"router.onnx", "expert_plug.onnx", "expert_light.onnx",
                    "expert_ac.onnx", "expert_water_heater.onnx"}
        actual = {f.name for f in onnx_files}
        assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"

        # Manifest
        assert (out_dir / "manifest.json").exists()
        print(f"  ✓ 5 ONNX files: {sorted(actual)}")
        print(f"  ✓ Total size: {manifest['total_size_bytes']/1024:.1f} KB")


def test_manifest_schema_valid():
    """Manifest has all required fields"""
    print("\n=== Test 4: Manifest Schema ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = _build_checkpoint(tmp)
        out_dir = tmp / "models"
        export_all(str(ckpt), str(out_dir))

        with open(out_dir / "manifest.json") as f:
            manifest = json.load(f)

        required_keys = {
            "version", "exported_at", "source_checkpoint",
            "opset_version", "categories", "input_spec",
            "configs", "artifacts", "total_size_bytes",
        }
        missing = required_keys - set(manifest.keys())
        assert not missing, f"Missing keys: {missing}"

        # Artifacts ครบ 5
        assert len(manifest["artifacts"]) == 5

        # ทุก artifact มี sha256
        for a in manifest["artifacts"]:
            assert "sha256" in a and len(a["sha256"]) == 64
        print(f"  ✓ Manifest schema complete: {len(required_keys)} required keys")
        print(f"  ✓ All 5 artifacts have SHA256 hashes")


def test_total_size_reasonable():
    """Total ONNX size < 50 MB for our model architecture"""
    print("\n=== Test 5: Total Size Reasonable ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = _build_checkpoint(tmp)
        out_dir = tmp / "models"
        manifest = export_all(str(ckpt), str(out_dir))

        total_mb = manifest["total_size_bytes"] / 1024 / 1024
        print(f"  Total ONNX size: {total_mb:.2f} MB")
        assert total_mb < 50, f"Total size too large: {total_mb} MB"


# ============================================================
# Verification tests
# ============================================================

def test_verify_router_passes_fresh_export():
    """Fresh export → verify passes for Router"""
    print("\n=== Test 6: Verify Router ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trainer = DRLSTFNTrainer(device="cpu")
        out_path = tmp / "router.onnx"
        export_router(
            trainer.router, out_path,
            seq_len=ROUTER_CONFIG.seq_len,
            n_features=ROUTER_CONFIG.n_features,
        )

        passed, results = verify_router(
            trainer.router, out_path,
            seq_len=ROUTER_CONFIG.seq_len,
            n_features=ROUTER_CONFIG.n_features,
            batch_sizes=[1, 4, 16],
        )
        assert passed, f"Router verify failed: {results}"
        max_abs_diff = max(r["max_abs_diff"] for r in results)
        print(f"  ✓ Verify passed, max_abs_diff across batches: {max_abs_diff:.2e}")


def test_verify_expert_passes_fresh_export():
    """Fresh export → verify passes for Expert"""
    print("\n=== Test 7: Verify Expert ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trainer = DRLSTFNTrainer(device="cpu")
        expert = trainer.experts["Plug"]
        out_path = tmp / "expert_plug.onnx"
        export_expert(
            expert, out_path, category="Plug",
            seq_len=EXPERT_CONFIG.seq_len,
            n_features=EXPERT_CONFIG.n_features,
        )

        passed, results = verify_expert(
            expert, out_path, category="Plug",
            seq_len=EXPERT_CONFIG.seq_len,
            n_features=EXPERT_CONFIG.n_features,
            batch_sizes=[1, 4, 16],
        )
        assert passed
        print(f"  ✓ Expert[Plug] verify passed across [1, 4, 16] batch sizes")


def test_verify_all_end_to_end():
    """Full export + verify pipeline"""
    print("\n=== Test 8: Verify All End-to-End ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = _build_checkpoint(tmp)
        out_dir = tmp / "models"
        export_all(str(ckpt), str(out_dir))

        passed, report = verify_all(str(ckpt), str(out_dir), batch_sizes=[1, 4])
        assert passed
        assert report["overall_passed"]

        # All 5 models reported
        assert "router" in report["models"]
        for cat in EXPERT_CONFIG.categories:
            assert f"expert_{cat}" in report["models"]
        print(f"  ✓ All 5 models verified passed")


def test_onnx_runtime_load():
    """Verify ONNX files load with onnxruntime + correct I/O metadata"""
    print("\n=== Test 9: ONNX Runtime Load ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt = _build_checkpoint(tmp)
        out_dir = tmp / "models"
        export_all(str(ckpt), str(out_dir))

        for onnx_file in out_dir.glob("*.onnx"):
            session = ort.InferenceSession(
                str(onnx_file), providers=["CPUExecutionProvider"]
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            assert len(inputs) == 1
            assert inputs[0].name == "features"
            assert len(outputs) == 1
        print(f"  ✓ All 5 ONNX files load successfully via onnxruntime")


def test_dynamic_batch_inference():
    """Test ONNX inference at multiple batch sizes (dynamic axes)"""
    print("\n=== Test 10: Dynamic Batch Inference ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trainer = DRLSTFNTrainer(device="cpu")
        out_path = tmp / "router.onnx"
        export_router(
            trainer.router, out_path,
            seq_len=ROUTER_CONFIG.seq_len,
            n_features=ROUTER_CONFIG.n_features,
        )

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

        for bs in [1, 7, 50, 100]:
            x = np.random.randn(
                bs, ROUTER_CONFIG.seq_len, ROUTER_CONFIG.n_features
            ).astype(np.float32)
            out = session.run(None, {"features": x})[0]
            assert out.shape[0] == bs, f"Batch size mismatch: {out.shape[0]} != {bs}"
        print(f"  ✓ Dynamic batch sizes work: [1, 7, 50, 100]")


if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Export Pipeline — Sanity Tests")
    print("=" * 60)

    test_router_export_creates_valid_onnx()
    test_expert_export_creates_valid_onnx()
    test_export_all_creates_5_files_plus_manifest()
    test_manifest_schema_valid()
    test_total_size_reasonable()
    test_verify_router_passes_fresh_export()
    test_verify_expert_passes_fresh_export()
    test_verify_all_end_to_end()
    test_onnx_runtime_load()
    test_dynamic_batch_inference()

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
