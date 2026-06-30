"""
CSV Logger — dependency-free metrics logging
=============================================
เขียน metrics ลง CSV ทุก step — สามารถเปิดด้วย Excel หรือ pandas

Features:
    - Auto-detect schema จาก first row
    - Append-only (resume training ได้)
    - Flush after each write (crash-safe)
    - Separate files per phase (pretrain.csv, rl.csv, eval.csv)
"""
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class CSVLogger:
    """
    Simple CSV logger ที่ append metrics row by row.
    
    Args:
        log_dir:     directory to write CSV files
        phase_name:  CSV filename = {phase_name}.csv
        flush_each:  flush ทันทีหลัง write (default True for crash-safety)
    
    Usage:
        >>> logger = CSVLogger("logs/run1", phase_name="pretrain")
        >>> logger.log({"step": 0, "loss": 1.23, "lr": 1e-3})
        >>> logger.log({"step": 10, "loss": 0.85, "lr": 1e-3})
        >>> logger.close()
    """
    def __init__(
        self,
        log_dir: str,
        phase_name: str = "metrics",
        flush_each: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.phase_name = phase_name
        self.flush_each = flush_each

        self.csv_path = self.log_dir / f"{phase_name}.csv"
        self._file = None
        self._writer = None
        self._fieldnames: Optional[List[str]] = None
        self._is_new_file = not self.csv_path.exists()

    def _ensure_open(self, fieldnames: List[str]):
        """Open file lazily; if existing file, read header."""
        if self._file is not None:
            return

        if self._is_new_file:
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._fieldnames = ["timestamp"] + fieldnames
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        else:
            # Append mode — read existing fieldnames
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                self._fieldnames = next(reader)
            self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)

    def log(self, metrics: Dict[str, Any]):
        """
        Append one row of metrics.
        
        Args:
            metrics: dict ของ metric name → value
                     (timestamp จะถูกเติมอัตโนมัติ)
        """
        # First call → discover schema
        if self._writer is None:
            self._ensure_open(list(metrics.keys()))

        row = {"timestamp": datetime.now(timezone.utc).isoformat()}
        for k in self._fieldnames:
            if k == "timestamp":
                continue
            v = metrics.get(k, "")
            # Convert numpy/torch scalars → python primitives
            if hasattr(v, "item"):
                v = v.item()
            row[k] = v
        self._writer.writerow(row)

        if self.flush_each:
            self._file.flush()

    def close(self):
        """Close file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MultiPhaseLogger:
    """
    Convenience wrapper — manages multiple CSV files for 3-phase curriculum.
    
    Usage:
        >>> log = MultiPhaseLogger("logs/run1")
        >>> log.pretrain.log({"step": 0, "loss": 1.5})
        >>> log.rl.log({"step": 0, "epsilon": 1.0, "loss/dqn": 0.5})
        >>> log.eval.log({"step": 100, "agg/teca": 0.85})
        >>> log.close_all()
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.pretrain = CSVLogger(log_dir, phase_name="pretrain")
        self.rl = CSVLogger(log_dir, phase_name="rl")
        self.eval = CSVLogger(log_dir, phase_name="eval")

    def close_all(self):
        self.pretrain.close()
        self.rl.close()
        self.eval.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
