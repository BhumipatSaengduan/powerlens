"""
Data Modules — abstract interface + implementations
====================================================
Decouple data source จาก trainer logic — switch ระหว่าง synthetic / real CSV
ได้โดยไม่แก้ training loop

Provides:
    - DataModule (abstract)
    - SyntheticDataModule (implementation จริง — ใช้ now)
    - CSVDataModule (real wide-CSV implementation)

Each module produces:
    - get_supervised_batch() → tensors สำหรับ Phase 1 pretrain
    - get_rl_step()          → single transition สำหรับ Phase 2 RL collect
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import numpy as np

from .synthetic import SyntheticGenerator, SyntheticSample


# ============================================================
# Data structures
# ============================================================

@dataclass
class SupervisedBatch:
    """Batch สำหรับ Phase 1 pretrain (supervised expert training)"""
    features: torch.Tensor                       # (B, T, F)
    truth_status: Dict[str, torch.Tensor]        # cat → (B, 1) — last-timestep label
    truth_power: Dict[str, torch.Tensor]         # cat → (B, 1)
    truth_current: Dict[str, torch.Tensor]       # cat → (B, 1)


@dataclass
class RLTransition:
    """Single transition สำหรับ Phase 2 RL"""
    state: torch.Tensor                          # (1, T, F)
    truth_active: torch.Tensor                   # (1, N) ∈ {0, 1}
    truth_power_per_cat: torch.Tensor            # (1, N) — for reward computation
    next_state: torch.Tensor                     # (1, T, F)
    done: float


# ============================================================
# Abstract base
# ============================================================

class DataModule(ABC):
    """
    Abstract data source for DRL-STFN training.
    
    Subclasses must implement get_supervised_batch() and get_rl_step().
    """
    def __init__(
        self,
        categories: list,
        device: str = "cpu",
        feature_columns: Optional[Sequence[str]] = None,
    ):
        self.categories = categories
        self.device = device
        self.feature_columns = list(feature_columns or [])

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @abstractmethod
    def get_supervised_batch(self, batch_size: int) -> SupervisedBatch:
        """Return one supervised batch สำหรับ Expert pretraining."""
        ...

    @abstractmethod
    def get_rl_step(self) -> RLTransition:
        """Return one transition สำหรับ Router RL training."""
        ...


# ============================================================
# Synthetic implementation
# ============================================================

class SyntheticDataModule(DataModule):
    """
    Synthetic data source — ใช้ SyntheticGenerator สร้างข้อมูลแบบ on-the-fly
    
    เหมาะสำหรับ:
        - Pipeline integration testing
        - Initial training ก่อนมี real data
        - Sanity check ว่า model converge ได้
    
    Args:
        window_size:   timesteps ต่อ window
        categories:    list of category names
        noise_level:   measurement noise
        seed:          RNG seed
        device:        torch device
    """
    def __init__(
        self,
        window_size: int = 60,
        categories: Optional[list] = None,
        noise_level: float = 0.02,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        cats = categories or ["Plug", "Light", "AC", "Water_Heater"]
        feature_columns = [
            "V_rms", "I_rms", "P", "Q", "PF", "THD",
            "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10",
        ]
        super().__init__(categories=cats, device=device, feature_columns=feature_columns)
        self.window_size = window_size
        self.generator = SyntheticGenerator(
            window_size=window_size,
            n_features=16,
            categories=cats,
            noise_level=noise_level,
            seed=seed,
        )
        # Cache one transition for RL "next state" continuity
        self._last_sample: Optional[SyntheticSample] = None

    def get_supervised_batch(self, batch_size: int) -> SupervisedBatch:
        """
        Generate batch + extract last-timestep labels for prediction.
        
        Note: model predicts state at last timestep ของ window, ดังนั้น label
        = ground truth at that timestep
        """
        batch = self.generator.generate_batch(batch_size)
        features = torch.from_numpy(batch.features).to(self.device)

        # Extract last-timestep labels per category
        truth_status = {}
        truth_power = {}
        truth_current = {}
        for cat in self.categories:
            # batch.truth_status[cat] has shape (B, T) → take last timestep
            s_last = batch.truth_status[cat][:, -1:]    # (B, 1)
            p_last = batch.truth_power[cat][:, -1:]
            c_last = batch.truth_current[cat][:, -1:]

            truth_status[cat] = torch.from_numpy(s_last).float().to(self.device)
            truth_power[cat] = torch.from_numpy(p_last).float().to(self.device)
            truth_current[cat] = torch.from_numpy(c_last).float().to(self.device)

        return SupervisedBatch(
            features=features,
            truth_status=truth_status,
            truth_power=truth_power,
            truth_current=truth_current,
        )

    def get_rl_step(self) -> RLTransition:
        """
        Generate single transition.
        
        For RL, "next_state" = next window. We chain windows ผ่าน cache.
        """
        # Current state
        if self._last_sample is None:
            current = self.generator.generate_window()
        else:
            current = self._last_sample

        # Next state
        next_sample = self.generator.generate_window()
        self._last_sample = next_sample

        # Convert to tensors (batch dim 1)
        state = torch.from_numpy(current.features).unsqueeze(0).to(self.device)
        next_state = torch.from_numpy(next_sample.features).unsqueeze(0).to(self.device)

        # Truth active mask at last timestep ของ current window — Router คำนวน reward
        # จาก state ปัจจุบัน, ดังนั้นใช้ active mask ของ current
        active_last = current.truth_active_mask[-1]    # (N,)
        truth_active = torch.from_numpy(active_last).long().unsqueeze(0).to(self.device)

        # Truth power per category at last timestep
        N = len(self.categories)
        power_last = np.zeros(N, dtype=np.float32)
        for i, cat in enumerate(self.categories):
            power_last[i] = current.truth_power[cat][-1]
        truth_power_per_cat = torch.from_numpy(power_last).unsqueeze(0).to(self.device)

        return RLTransition(
            state=state,
            truth_active=truth_active,
            truth_power_per_cat=truth_power_per_cat,
            next_state=next_state,
            done=0.0,    # synthetic: never terminates
        )


# ============================================================
# CSV implementation
# ============================================================

class CSVDataModule(DataModule):
    """
    Real wide-CSV data source for all-feature DRL-STFN training.

    Expected wide schema:
        timestamp, feature_1, feature_2, ..., AC_status, AC_power, AC_current, ...

    Notes:
        - If feature_columns is omitted, all numeric non-target columns are used.
        - Categories are inferred from columns ending with status_suffix if omitted.
        - Missing feature values are linearly interpolated per column.
        - Robust scaling is fit on the chronological train split only.
        - This does not pivot long per-device CSVs. Prepare those into a wide
          aggregate table first if needed.
    """
    DEFAULT_EXCLUDED_COLUMNS = {
        "timestamp", "date", "datetime", "time",
        "site_id", "device_id", "mqtt_topic", "aws_thing_name", "serial_no", "label",
        "device", "category", "train_set", "split",
        "checksum32", "chunk_seq",
    }

    def __init__(
        self,
        csv_path: str,
        categories: Optional[Sequence[str]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        timestamp_col: str = "timestamp",
        status_suffix: str = "_status",
        power_suffix: str = "_power",
        current_suffix: str = "_current",
        window_size: int = 60,
        train: bool = True,
        train_ratio: float = 0.8,
        scale: bool = True,
        scaler_state: Optional[dict] = None,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        self.csv_path = str(csv_path)
        self.timestamp_col = timestamp_col
        self.status_suffix = status_suffix
        self.power_suffix = power_suffix
        self.current_suffix = current_suffix
        self.window_size = window_size
        self.train = train
        self.train_ratio = train_ratio
        self.scale = scale
        self.rng = np.random.default_rng(seed)

        rows, fieldnames = self._read_rows(self.csv_path)
        if len(rows) < window_size + 1:
            raise ValueError(
                f"CSV has {len(rows)} rows, but window_size={window_size} needs at least {window_size + 1}"
            )

        inferred_categories = list(categories or self._infer_categories(fieldnames, status_suffix))
        if not inferred_categories:
            raise ValueError(
                "No categories found. Pass --categories or provide columns like '<category>_status'."
            )

        target_columns = self._target_columns(
            inferred_categories, status_suffix, power_suffix, current_suffix
        )
        selected_features = list(feature_columns or self._infer_feature_columns(
            rows, fieldnames, target_columns, timestamp_col,
        ))
        if not selected_features:
            raise ValueError("No feature columns selected or inferred from CSV.")

        super().__init__(
            categories=inferred_categories,
            device=device,
            feature_columns=selected_features,
        )

        raw_features = self._columns_to_matrix(rows, selected_features)
        raw_features = self._fill_missing(raw_features)

        self.n_rows = raw_features.shape[0]
        split_idx = int(self.n_rows * train_ratio)
        split_idx = min(max(split_idx, window_size), self.n_rows - 1)
        self.split_idx = split_idx

        train_features = raw_features[:split_idx]
        if scale:
            self.scaler_state = scaler_state or self._fit_robust_scaler(train_features)
            self.features = self._apply_robust_scaler(raw_features, self.scaler_state)
        else:
            self.scaler_state = {
                "method": "none",
                "feature_columns": selected_features,
                "center": [0.0 for _ in selected_features],
                "scale": [1.0 for _ in selected_features],
            }
            self.features = raw_features.astype(np.float32)

        self.truth_status = {}
        self.truth_power = {}
        self.truth_current = {}
        self.missing_power_targets = []
        for cat in self.categories:
            s_col = f"{cat}{status_suffix}"
            p_col = f"{cat}{power_suffix}"
            c_col = f"{cat}{current_suffix}"

            self.truth_status[cat] = self._column_to_float(rows, s_col, required=True)
            if p_col in fieldnames:
                self.truth_power[cat] = self._column_to_float(rows, p_col, required=False)
            else:
                self.truth_power[cat] = np.zeros(self.n_rows, dtype=np.float32)
                self.missing_power_targets.append(cat)
            self.truth_current[cat] = self._column_to_float(rows, c_col, required=True)

        self.valid_starts = self._build_window_starts(train=train)
        if len(self.valid_starts) == 0:
            split_name = "train" if train else "validation"
            raise ValueError(f"No valid {split_name} windows for window_size={window_size}")

    def get_supervised_batch(self, batch_size: int) -> SupervisedBatch:
        starts = self._sample_starts(batch_size)
        features = self._windows_from_starts(starts)
        target_idx = starts + self.window_size - 1

        truth_status = {}
        truth_power = {}
        truth_current = {}
        for cat in self.categories:
            truth_status[cat] = self._target_tensor(self.truth_status[cat], target_idx)
            truth_power[cat] = self._target_tensor(self.truth_power[cat], target_idx)
            truth_current[cat] = self._target_tensor(self.truth_current[cat], target_idx)

        return SupervisedBatch(
            features=torch.from_numpy(features).to(self.device),
            truth_status=truth_status,
            truth_power=truth_power,
            truth_current=truth_current,
        )

    def get_rl_step(self) -> RLTransition:
        start = int(self._sample_starts(1)[0])
        next_start = min(start + 1, self.n_rows - self.window_size)

        state = self.features[start:start + self.window_size][None, :, :].astype(np.float32)
        next_state = self.features[next_start:next_start + self.window_size][None, :, :].astype(np.float32)

        target_idx = start + self.window_size - 1
        active = np.array([
            1 if self.truth_status[cat][target_idx] >= 0.5 else 0
            for cat in self.categories
        ], dtype=np.int64)[None, :]
        power = np.array([
            self.truth_power[cat][target_idx]
            for cat in self.categories
        ], dtype=np.float32)[None, :]

        return RLTransition(
            state=torch.from_numpy(state).to(self.device),
            truth_active=torch.from_numpy(active).to(self.device),
            truth_power_per_cat=torch.from_numpy(power).to(self.device),
            next_state=torch.from_numpy(next_state).to(self.device),
            done=0.0,
        )

    def preprocessing_manifest(self) -> dict:
        """Metadata that must travel with exported models to AWS Apply AI."""
        return {
            "source_csv": str(Path(self.csv_path).resolve()),
            "format": "wide_csv_sequence_to_one",
            "timestamp_col": self.timestamp_col,
            "feature_columns": self.feature_columns,
            "n_features": self.n_features,
            "categories": self.categories,
            "window_size": self.window_size,
            "target_suffixes": {
                "status": self.status_suffix,
                "power": self.power_suffix,
                "current": self.current_suffix,
            },
            "train_ratio": self.train_ratio,
            "split_idx": self.split_idx,
            "scale": self.scale,
            "scaler": self.scaler_state,
            "missing_power_targets_filled_with_zero": self.missing_power_targets,
        }

    @staticmethod
    def _read_rows(csv_path: str) -> Tuple[List[dict], List[str]]:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if not fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        return rows, fieldnames

    @staticmethod
    def _infer_categories(fieldnames: Sequence[str], status_suffix: str) -> List[str]:
        cats = []
        for col in fieldnames:
            if col.endswith(status_suffix):
                cats.append(col[:-len(status_suffix)])
        return cats

    @classmethod
    def _infer_feature_columns(
        cls,
        rows: Sequence[dict],
        fieldnames: Sequence[str],
        target_columns: set,
        timestamp_col: str,
    ) -> List[str]:
        excluded = set(cls.DEFAULT_EXCLUDED_COLUMNS)
        excluded.add(timestamp_col)
        excluded.update(target_columns)

        feature_columns = []
        for col in fieldnames:
            if col in excluded:
                continue
            values = cls._column_to_float(rows, col, required=False)
            valid_ratio = float(np.isfinite(values).mean())
            if valid_ratio > 0.5:
                feature_columns.append(col)
        return feature_columns

    @staticmethod
    def _target_columns(
        categories: Sequence[str],
        status_suffix: str,
        power_suffix: str,
        current_suffix: str,
    ) -> set:
        cols = set()
        for cat in categories:
            cols.add(f"{cat}{status_suffix}")
            cols.add(f"{cat}{power_suffix}")
            cols.add(f"{cat}{current_suffix}")
        return cols

    @staticmethod
    def _column_to_float(
        rows: Sequence[dict],
        col: str,
        required: bool = False,
    ) -> np.ndarray:
        values = np.empty(len(rows), dtype=np.float32)
        missing_col = len(rows) > 0 and col not in rows[0]
        if missing_col and required:
            raise ValueError(f"Required column missing: {col}")
        if missing_col:
            values.fill(np.nan)
            return values

        for i, row in enumerate(rows):
            raw = row.get(col, "")
            if raw is None or raw == "":
                values[i] = np.nan
                continue
            try:
                values[i] = float(raw)
            except ValueError:
                values[i] = np.nan
        return values

    @classmethod
    def _columns_to_matrix(cls, rows: Sequence[dict], columns: Sequence[str]) -> np.ndarray:
        arrays = [cls._column_to_float(rows, col, required=True) for col in columns]
        return np.stack(arrays, axis=1).astype(np.float32)

    @staticmethod
    def _fill_missing(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=True)
        idx = np.arange(x.shape[0])
        for j in range(x.shape[1]):
            col = x[:, j]
            valid = np.isfinite(col)
            if valid.all():
                continue
            if valid.sum() == 0:
                col[:] = 0.0
            elif valid.sum() == 1:
                col[:] = col[valid][0]
            else:
                col[~valid] = np.interp(idx[~valid], idx[valid], col[valid])
            x[:, j] = col
        return x

    @staticmethod
    def _fit_robust_scaler(x_train: np.ndarray) -> dict:
        center = np.nanmedian(x_train, axis=0)
        q1 = np.nanpercentile(x_train, 25, axis=0)
        q3 = np.nanpercentile(x_train, 75, axis=0)
        scale = q3 - q1
        scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
        return {
            "method": "robust",
            "center": center.astype(float).tolist(),
            "scale": scale.astype(float).tolist(),
        }

    @staticmethod
    def _apply_robust_scaler(x: np.ndarray, scaler_state: dict) -> np.ndarray:
        center = np.asarray(scaler_state["center"], dtype=np.float32)
        scale = np.asarray(scaler_state["scale"], dtype=np.float32)
        return ((x - center) / scale).astype(np.float32)

    def _build_window_starts(self, train: bool) -> np.ndarray:
        target_indices = np.arange(self.window_size - 1, self.n_rows)
        if train:
            target_indices = target_indices[target_indices < self.split_idx]
        else:
            target_indices = target_indices[target_indices >= self.split_idx]
        starts = target_indices - (self.window_size - 1)
        starts = starts[starts + self.window_size <= self.n_rows]
        return starts.astype(np.int64)

    def _sample_starts(self, batch_size: int) -> np.ndarray:
        replace = len(self.valid_starts) < batch_size
        return self.rng.choice(self.valid_starts, size=batch_size, replace=replace)

    def _windows_from_starts(self, starts: np.ndarray) -> np.ndarray:
        windows = [
            self.features[start:start + self.window_size]
            for start in starts
        ]
        return np.stack(windows, axis=0).astype(np.float32)

    def _target_tensor(self, values: np.ndarray, target_idx: np.ndarray) -> torch.Tensor:
        target = values[target_idx].reshape(-1, 1).astype(np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(target).to(self.device)
