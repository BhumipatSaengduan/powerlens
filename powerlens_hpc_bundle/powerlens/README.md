# PowerLens HPC Bundle

DRL-STFN — Hierarchical NILM disaggregation system
HPC training side (เสร็จแล้ว ณ วันที่บันทึก)

## โครงสร้างไฟล์

```
powerlens/
├── models/         (10 ไฟล์)  Architecture + algorithm
│   ├── config.py            # 9 dataclass configs
│   ├── sftn.py              # SFTN feature extraction
│   ├── expert.py            # Stage 2 Expert × 4 categories
│   ├── router.py            # Stage 1 Branching DQN
│   ├── replay_buffer.py     # PER + Uniform
│   ├── reward.py            # per-head reward (TP/TN/FP/FN)
│   ├── trainer.py           # 3-phase orchestrator
│   ├── inference.py         # Stage 3 confidence + constraint + retry
│   └── fusion.py            # Stage 4 rebalance + residual + JSON
│
├── data/           (3 ไฟล์)   Data sources
│   ├── synthetic.py         # generator 16 features realistic profiles
│   └── data_module.py       # SyntheticDataModule + CSVDataModule
│
├── training/       (4 ไฟล์)   Training infrastructure
│   ├── train.py             # CLI entrypoint
│   ├── evaluator.py         # per-cat + Router + TECA metrics
│   └── csv_logger.py        # dependency-free logging
│
├── deployment/     (3 ไฟล์)   ONNX export
│   ├── export.py            # PyTorch → 5 ONNX files
│   └── verify.py            # numerical equivalence
│
└── tests/          (8 ไฟล์)   81 tests, 7 suites — ผ่านครบ
```

## วิธีใช้

### Train
```bash
python -m powerlens.training.train \
    --output-dir runs/exp1 \
    --pretrain-steps 5000 \
    --rl-steps 20000 \
    --batch-size 64 \
    --eval-every 500 \
    --device cuda \
    --seed 42
```

### Train with real all-feature CSV
```bash
python -m powerlens.training.train \
    --data-csv data/processed/train_wide.csv \
    --output-dir runs/all_features_v1 \
    --seq-len 60 \
    --pretrain-steps 5000 \
    --rl-steps 20000 \
    --batch-size 64 \
    --loss-weight-power 0 \
    --device cuda
```

If `--feature-cols` is omitted, the trainer uses all numeric non-target columns
as input features. Targets are inferred from columns like
`AC_status`, `AC_current`, and optional `AC_power`.

### Export ONNX
```bash
python -m powerlens.deployment.export \
    --checkpoint runs/exp1/checkpoints/best.pt \
    --output-dir models/drl_stfn_v1/
```

### Verify
```bash
python -m powerlens.deployment.verify \
    --checkpoint runs/exp1/checkpoints/best.pt \
    --onnx-dir models/drl_stfn_v1/
```

### Run tests
```bash
for t in test_expert test_router test_training test_inference test_fusion test_training_infra test_onnx_export; do
    python powerlens/tests/$t.py
done
```

## Dependencies

- torch >= 2.0
- numpy
- onnx
- onnxruntime
- onnxscript

## Status

- 🟢 HPC training pipeline complete
- 🟢 ONNX export pipeline complete
- 🔴 EC2 inference wrapper — pending
- 🟡 Refactor + GitHub structure — pending
