"""
Synthetic Data Generator for DRL-STFN Training
================================================
สร้าง synthetic appliance signatures สำหรับ train pipeline ก่อนมี real sensor data

Design:
    - 4 categories: Plug, Light, AC, Water_Heater
    - แต่ละ category มี realistic signature distinguishable จากกัน
    - 16 features ตาม spec: V_rms, I_rms, P, Q, PF, THD, H1-H10
    - ground-truth labels ครบ: status, power, current per category

Appliance signatures (per-category templates):
    Plug      → small steady load (laptop/phone charger)
                P ≈ 5-100W, low harmonics, PF ≈ 0.95-1.0
    Light     → very small load, often LED
                P ≈ 3-50W, moderate H3 (LED driver), PF ≈ 0.9-0.95
    AC        → cycling compressor (induction motor)
                P ≈ 500-2000W, on/off cycles, high startup current,
                significant H3/H5 (motor harmonics), PF ≈ 0.7-0.85
    Water_Heater → resistive load, simple on/off
                P ≈ 1000-3000W, very low harmonics, PF ≈ 0.95-1.0

Aggregate construction:
    aggregate_features = sum(per-category contributions when active) + noise

Usage:
    >>> gen = SyntheticGenerator(window_size=60)
    >>> sample = gen.generate_window()
    >>> # sample.features: (60, 16), sample.labels: dict per category
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================
# Per-category profile constants
# ============================================================

@dataclass
class CategoryProfile:
    """Realistic signature parameters per category."""
    name: str
    # Power range (W) when active
    power_min: float
    power_max: float
    # Power factor range
    pf_min: float
    pf_max: float
    # Harmonic content (relative magnitudes for H1-H10, normalized)
    # H1 = fundamental (always 1.0), others = harmonic distortion ratios
    harmonic_ratios: Tuple[float, ...]
    # Cycling pattern: probability of being "on" at any random moment
    duty_cycle: float
    # Within "on" period: do we cycle? (e.g. AC compressor)
    cycles_within_on: bool
    # Voltage at this site (V) — สมมติ 220V Thai grid
    voltage: float = 220.0


# Realistic profiles — calibrated จาก typical commercial loads
PROFILES: Dict[str, CategoryProfile] = {
    "Plug": CategoryProfile(
        name="Plug",
        power_min=5, power_max=80,
        pf_min=0.95, pf_max=1.0,
        harmonic_ratios=(1.0, 0.02, 0.05, 0.01, 0.03, 0.005, 0.01, 0.005, 0.005, 0.002),
        duty_cycle=0.6,
        cycles_within_on=False,
    ),
    "Light": CategoryProfile(
        name="Light",
        power_min=3, power_max=50,
        pf_min=0.85, pf_max=0.95,
        # LED driver มี H3 prominent
        harmonic_ratios=(1.0, 0.05, 0.20, 0.03, 0.10, 0.02, 0.05, 0.01, 0.02, 0.005),
        duty_cycle=0.7,
        cycles_within_on=False,
    ),
    "AC": CategoryProfile(
        name="AC",
        power_min=500, power_max=2000,
        pf_min=0.70, pf_max=0.85,
        # Induction motor: high reactive power, H3/H5 prominent
        harmonic_ratios=(1.0, 0.08, 0.15, 0.05, 0.12, 0.03, 0.06, 0.02, 0.03, 0.01),
        duty_cycle=0.4,
        cycles_within_on=True,    # compressor cycle
    ),
    "Water_Heater": CategoryProfile(
        name="Water_Heater",
        power_min=1000, power_max=3000,
        pf_min=0.95, pf_max=1.0,
        # Pure resistive — minimal harmonics
        harmonic_ratios=(1.0, 0.01, 0.02, 0.005, 0.01, 0.005, 0.005, 0.002, 0.002, 0.001),
        duty_cycle=0.2,
        cycles_within_on=False,
    ),
}


# ============================================================
# Output schema
# ============================================================

@dataclass
class SyntheticSample:
    """One generated training sample"""
    features: np.ndarray          # (T, 16) — aggregate features
    # Per-category ground truth (each shape (T,))
    truth_status: Dict[str, np.ndarray]    # binary {0, 1}
    truth_power: Dict[str, np.ndarray]     # Watts
    truth_current: Dict[str, np.ndarray]   # Amperes
    truth_active_mask: np.ndarray          # (T, N_categories) — overall any-active per timestep


# ============================================================
# Generator
# ============================================================

class SyntheticGenerator:
    """
    Generate synthetic NILM training data.
    
    Strategy per window:
        1. Decide which categories are active in this window (via duty_cycle)
        2. For each active category, generate per-timestep status + power + current
           - Resistive: stable on/off
           - Cycling (AC): compressor on/off cycle within window
        3. Compute per-category contribution to features (V_rms shared, I from sum)
        4. Combine into aggregate features + add measurement noise
        5. Compute aggregate labels (truth_*)
    
    Args:
        window_size:     timesteps per sample (default 60 = 1 minute @ 1 Hz)
        n_features:      total feature dim (must be 16 for current architecture)
        categories:      list of category names (must match PROFILES keys)
        noise_level:     gaussian noise σ relative to feature magnitude
        seed:            optional RNG seed for reproducibility
    """
    def __init__(
        self,
        window_size: int = 60,
        n_features: int = 16,
        categories: Optional[List[str]] = None,
        noise_level: float = 0.02,
        seed: Optional[int] = None,
    ):
        if n_features != 16:
            raise ValueError("Generator currently outputs 16 features only")
        self.window_size = window_size
        self.n_features = n_features
        self.categories = categories or list(PROFILES.keys())
        # Verify all requested categories have profiles
        for cat in self.categories:
            if cat not in PROFILES:
                raise ValueError(f"No profile for category '{cat}'")
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    # Per-category trace generation
    # ----------------------------------------------------------------
    def _generate_category_trace(
        self, profile: CategoryProfile
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate (status, power, current) traces for one category over window.
        
        Returns:
            status:  (T,) ∈ {0, 1}
            power:   (T,) Watts
            current: (T,) Amperes
        """
        T = self.window_size

        # Decide if category is "available" this window
        is_available = self.rng.random() < profile.duty_cycle
        if not is_available:
            return np.zeros(T), np.zeros(T), np.zeros(T)

        # Sample power level for this window
        power_level = self.rng.uniform(profile.power_min, profile.power_max)
        pf = self.rng.uniform(profile.pf_min, profile.pf_max)

        # Status pattern
        if profile.cycles_within_on:
            # Compressor-like: cycles on/off within window
            # Period ~10-30 timesteps, duty ~50-70%
            period = self.rng.integers(10, 30)
            duty = self.rng.uniform(0.5, 0.7)
            phase = self.rng.integers(0, period)
            t_idx = (np.arange(T) + phase) % period
            status = (t_idx < period * duty).astype(np.float32)
        else:
            # Steady on/off — stays on for 60-100% of window
            on_duration = int(T * self.rng.uniform(0.6, 1.0))
            start = self.rng.integers(0, max(1, T - on_duration + 1))
            status = np.zeros(T, dtype=np.float32)
            status[start:start + on_duration] = 1.0

        # Power = power_level × status, with small variation
        variation = 1.0 + 0.05 * self.rng.standard_normal(T)
        power = power_level * status * variation
        # Apparent power = P / PF
        apparent_power = power / max(pf, 1e-3)
        # Current = S / V (RMS)
        current = apparent_power / profile.voltage

        return status, power, current

    # ----------------------------------------------------------------
    # Feature aggregation
    # ----------------------------------------------------------------
    def _compute_features(
        self,
        per_category_power: Dict[str, np.ndarray],
        per_category_current: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Combine per-category contributions → aggregate 16-feature vector per timestep.
        
        Features (in order):
            [V_rms, I_rms, P, Q, PF, THD, H1, H2, ..., H10]
        """
        T = self.window_size
        features = np.zeros((T, 16), dtype=np.float32)

        # V_rms — assume nominal 220V with small jitter
        v_rms = 220.0 + 2.0 * self.rng.standard_normal(T)
        features[:, 0] = v_rms

        # Aggregate I_rms (sum of per-category currents)
        i_total = np.zeros(T)
        for cat in self.categories:
            i_total += per_category_current[cat]
        features[:, 1] = i_total

        # Aggregate P (active power)
        p_total = np.zeros(T)
        for cat in self.categories:
            p_total += per_category_power[cat]
        features[:, 2] = p_total

        # Aggregate apparent power S = V × I
        s_total = v_rms * i_total
        # Q = sqrt(S² - P²) if S > P else 0
        q_squared = np.maximum(s_total**2 - p_total**2, 0)
        q_total = np.sqrt(q_squared)
        features[:, 3] = q_total

        # PF = P / S (avoid div by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            pf_total = np.where(s_total > 1e-3, p_total / s_total, 0.0)
        features[:, 4] = np.clip(np.nan_to_num(pf_total, nan=0.0), 0.0, 1.0)

        # THD and H1-H10 — weighted average ของ per-category harmonics
        # โดย weight ตาม current contribution
        h_aggregate = np.zeros((T, 10), dtype=np.float32)
        for cat in self.categories:
            profile = PROFILES[cat]
            i_cat = per_category_current[cat]
            # Weight = i_cat / i_total (per timestep)
            weight = np.where(i_total > 1e-6, i_cat / np.maximum(i_total, 1e-6), 0.0)
            for h_idx in range(10):
                h_aggregate[:, h_idx] += weight * profile.harmonic_ratios[h_idx]

        # THD = sqrt(Σ Hi² for i≥2) / H1
        thd_num = np.sqrt(np.sum(h_aggregate[:, 1:]**2, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            thd = np.where(h_aggregate[:, 0] > 1e-6, thd_num / h_aggregate[:, 0], 0.0)
        features[:, 5] = np.nan_to_num(thd, nan=0.0)

        # H1-H10 (raw magnitudes — model จะ normalize เอง)
        features[:, 6:16] = h_aggregate

        # Apply measurement noise
        if self.noise_level > 0:
            noise = self.rng.standard_normal(features.shape).astype(np.float32)
            magnitude = np.abs(features) + 1.0
            features += self.noise_level * noise * magnitude

        return features

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def generate_window(self) -> SyntheticSample:
        """Generate one full sample (one window)."""
        per_status: Dict[str, np.ndarray] = {}
        per_power: Dict[str, np.ndarray] = {}
        per_current: Dict[str, np.ndarray] = {}

        for cat in self.categories:
            s, p, c = self._generate_category_trace(PROFILES[cat])
            per_status[cat] = s.astype(np.float32)
            per_power[cat] = p.astype(np.float32)
            per_current[cat] = c.astype(np.float32)

        features = self._compute_features(per_power, per_current)

        # Aggregate any-active mask: shape (T, N) — สำหรับ Router action ground truth
        truth_active_mask = np.stack(
            [per_status[cat] for cat in self.categories], axis=-1
        ).astype(np.float32)

        return SyntheticSample(
            features=features,
            truth_status=per_status,
            truth_power=per_power,
            truth_current=per_current,
            truth_active_mask=truth_active_mask,
        )

    def generate_batch(self, batch_size: int) -> SyntheticSample:
        """
        Generate batch of windows — stacked.
        
        Returns SyntheticSample where:
            features:           (B, T, 16)
            truth_status:       dict[cat → (B, T)]
            truth_power:        dict[cat → (B, T)]
            truth_current:      dict[cat → (B, T)]
            truth_active_mask:  (B, T, N)
        """
        samples = [self.generate_window() for _ in range(batch_size)]
        # Stack
        features = np.stack([s.features for s in samples], axis=0)
        truth_active_mask = np.stack([s.truth_active_mask for s in samples], axis=0)

        truth_status = {
            cat: np.stack([s.truth_status[cat] for s in samples], axis=0)
            for cat in self.categories
        }
        truth_power = {
            cat: np.stack([s.truth_power[cat] for s in samples], axis=0)
            for cat in self.categories
        }
        truth_current = {
            cat: np.stack([s.truth_current[cat] for s in samples], axis=0)
            for cat in self.categories
        }

        return SyntheticSample(
            features=features,
            truth_status=truth_status,
            truth_power=truth_power,
            truth_current=truth_current,
            truth_active_mask=truth_active_mask,
        )
