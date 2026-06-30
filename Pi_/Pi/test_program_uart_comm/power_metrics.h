// ============================================================
// power_metrics.h
// ------------------------------------------------------------
// Electrical metric computation on Pi Pico 2W
//   from ring buffer of raw V(t), I(t) int16 samples
//   to: Vrms, Irms, P, S, PF, frequency, harmonic, phase, ...
//
// Algorithm: Goertzel single-bin DFT (chosen over full FFT
//   because we know target frequencies — 50 Hz and harmonics)
//
// Channel layout in flat sample buffer (matches the .ino):
//   index 0 = V1   index 1 = I1
//   index 2 = V2   index 3 = I2
//   index 4 = V3   index 5 = I3
//
// Author: PowerLens
// ============================================================
#ifndef POWER_METRICS_H
#define POWER_METRICS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define N_PHASES        3
#define N_HARMONICS     7        // 2nd..7th for THD (1st is fundamental)
#define LINE_FREQ_HZ    50.0f    // Thailand grid

// ── Calibration ─────────────────────────────────────────────
// Original .ino stores ADC samples as int16 where:
//   adc_volts = int16_value / 5000.0
//
// Field calibration workflow:
//   new V scale = old V scale * (meter_Vrms / Pi_Vrms)
//   new I scale = old I scale * (meter_Irms / Pi_Irms)
//
// To keep that workflow explicit in code, we split each channel into:
//   1) a base placeholder scale from bench bring-up
//   2) a reference pair (actual meter reading vs Pi reading)
//
// Leave the reference pair at 1.0 / 1.0 for the current behavior.
// After field measurement, update only the *_REF_ACTUAL_* and
// *_REF_OBSERVED_* values and reflash the Pi.
#ifndef SCALE_FROM_REFERENCE
#define SCALE_FROM_REFERENCE(base_scale, actual_value, observed_value) \
    ((base_scale) * (((observed_value) != 0.0f) ? ((actual_value) / (observed_value)) : 1.0f))
#endif

#ifndef V_SCALE_BASE_PHASE1
#define V_SCALE_BASE_PHASE1   0.0785904165665054f
#endif
#ifndef V_SCALE_BASE_PHASE2
#define V_SCALE_BASE_PHASE2   0.0785904165665054f
#endif
#ifndef V_SCALE_BASE_PHASE3
#define V_SCALE_BASE_PHASE3   0.0785904165665054f
#endif

#ifndef I_SCALE_BASE_PHASE1
#define I_SCALE_BASE_PHASE1   0.000926820893136259f
#endif
#ifndef I_SCALE_BASE_PHASE2
#define I_SCALE_BASE_PHASE2   0.000926820893136259f
#endif
#ifndef I_SCALE_BASE_PHASE3
#define I_SCALE_BASE_PHASE3   0.000926820893136259f
#endif

#ifndef V_REF_ACTUAL_PHASE1
#define V_REF_ACTUAL_PHASE1   1.0f
#endif
#ifndef V_REF_ACTUAL_PHASE2
#define V_REF_ACTUAL_PHASE2   1.0f
#endif
#ifndef V_REF_ACTUAL_PHASE3
#define V_REF_ACTUAL_PHASE3   1.0f
#endif

#ifndef V_REF_OBSERVED_PHASE1
#define V_REF_OBSERVED_PHASE1 1.0f
#endif
#ifndef V_REF_OBSERVED_PHASE2
#define V_REF_OBSERVED_PHASE2 1.0f
#endif
#ifndef V_REF_OBSERVED_PHASE3
#define V_REF_OBSERVED_PHASE3 1.0f
#endif

#ifndef I_REF_ACTUAL_PHASE1
#define I_REF_ACTUAL_PHASE1   0.0503f
#endif
#ifndef I_REF_ACTUAL_PHASE2
#define I_REF_ACTUAL_PHASE2   0.0503f
#endif
#ifndef I_REF_ACTUAL_PHASE3
#define I_REF_ACTUAL_PHASE3   0.0503f
#endif

#ifndef I_REF_OBSERVED_PHASE1
#define I_REF_OBSERVED_PHASE1 5.08f
#endif
#ifndef I_REF_OBSERVED_PHASE2
#define I_REF_OBSERVED_PHASE2 5.08f
#endif
#ifndef I_REF_OBSERVED_PHASE3
#define I_REF_OBSERVED_PHASE3 5.08f
#endif

#ifndef V_SCALE_PHASE1
#define V_SCALE_PHASE1   SCALE_FROM_REFERENCE(V_SCALE_BASE_PHASE1, V_REF_ACTUAL_PHASE1, V_REF_OBSERVED_PHASE1)
#endif
#ifndef V_SCALE_PHASE2
#define V_SCALE_PHASE2   SCALE_FROM_REFERENCE(V_SCALE_BASE_PHASE2, V_REF_ACTUAL_PHASE2, V_REF_OBSERVED_PHASE2)
#endif
#ifndef V_SCALE_PHASE3
#define V_SCALE_PHASE3   SCALE_FROM_REFERENCE(V_SCALE_BASE_PHASE3, V_REF_ACTUAL_PHASE3, V_REF_OBSERVED_PHASE3)
#endif

// Legacy linear current scale. Kept as a fallback when
// I_CT_NONLINEAR_ENABLE is set to 0.
#ifndef I_SCALE_PHASE1
#define I_SCALE_PHASE1   SCALE_FROM_REFERENCE(I_SCALE_BASE_PHASE1, I_REF_ACTUAL_PHASE1, I_REF_OBSERVED_PHASE1)
#endif
#ifndef I_SCALE_PHASE2
#define I_SCALE_PHASE2   SCALE_FROM_REFERENCE(I_SCALE_BASE_PHASE2, I_REF_ACTUAL_PHASE2, I_REF_OBSERVED_PHASE2)
#endif
#ifndef I_SCALE_PHASE3
#define I_SCALE_PHASE3   SCALE_FROM_REFERENCE(I_SCALE_BASE_PHASE3, I_REF_ACTUAL_PHASE3, I_REF_OBSERVED_PHASE3)
#endif

// ── Current transfer function ───────────────────────────────
// Clamp-meter transfer curve from latest calibration data:
//   i_A = 30.0 * (V_formula - offset)^1.000
//
// The ADC current channel is reconstructed from raw int16 as:
//   V_adc ≈ raw_I / 5000.0
//
// We intentionally apply the curve at the 1-second window level:
//   1) center the current-channel ADC waveform
//   2) measure one statistic of that centered waveform
//      (mean-abs, RMS, or peak; selected below)
//   3) convert that statistic to Irms with the fitted curve
//   4) rescale the centered current waveform to that Irms so P/PF/phase still
//      use the actual waveform shape.
#ifndef I_CT_NONLINEAR_ENABLE
#define I_CT_NONLINEAR_ENABLE   1
#endif

#ifndef I_CT_AUTO_ZERO_OFFSET
#define I_CT_AUTO_ZERO_OFFSET   1
#endif

#ifndef I_CT_K
#define I_CT_K                  30.0f
#endif

#ifndef I_CT_EXP
#define I_CT_EXP                1.0f
#endif

#ifndef I_CT_EPSILON_V
#define I_CT_EPSILON_V          0.0001f
#endif

#ifndef I_CT_MAX_A
#define I_CT_MAX_A              50.0f
#endif

// Statistic to feed into the current transfer curve.
// RMS is the safest production default because it is much less sensitive to
// isolated ADC spikes than peak mode. If a future controlled calibration proves
// the professor's equation expects peak/sqrt(2), switch this back to PEAK.
#define I_CT_STAT_MODE_MEAN_ABS 1
#define I_CT_STAT_MODE_RMS      2
#define I_CT_STAT_MODE_PEAK     3

#ifndef I_CT_STAT_MODE
#define I_CT_STAT_MODE          I_CT_STAT_MODE_RMS
#endif

// When peak mode is selected, convert peak amplitude to the quantity expected
// by the transfer curve. Default behavior is peak-to-RMS-equivalent.
#ifndef I_CT_PEAK_TO_FORMULA_DIVISOR
#define I_CT_PEAK_TO_FORMULA_DIVISOR 1.41421356f
#endif

// Ignore very small ADC-side current motion as noise. 40 mV class signals
// still pass; millivolt-level no-load jitter becomes 0 A instead of fake amps.
#ifndef I_CT_FORMULA_DEADBAND_V
#define I_CT_FORMULA_DEADBAND_V 0.005f
#endif

// Zero-current floor in the formula-voltage domain. This offset is applied to
// whichever statistic is selected above. Start at 0 and raise only if no-load
// current still sits too high after wiring is corrected.
#ifndef I_CT_ADC_MEAN_OFFSET_V_PHASE1
#define I_CT_ADC_MEAN_OFFSET_V_PHASE1 0.0f
#endif
#ifndef I_CT_ADC_MEAN_OFFSET_V_PHASE2
#define I_CT_ADC_MEAN_OFFSET_V_PHASE2 0.0f
#endif
#ifndef I_CT_ADC_MEAN_OFFSET_V_PHASE3
#define I_CT_ADC_MEAN_OFFSET_V_PHASE3 0.0f
#endif

// Backward-compatible alias: older configs used the "MEAN" name even when the
// formula really wants a different statistic.
#ifndef I_CT_ADC_FORMULA_OFFSET_V_PHASE1
#define I_CT_ADC_FORMULA_OFFSET_V_PHASE1 I_CT_ADC_MEAN_OFFSET_V_PHASE1
#endif
#ifndef I_CT_ADC_FORMULA_OFFSET_V_PHASE2
#define I_CT_ADC_FORMULA_OFFSET_V_PHASE2 I_CT_ADC_MEAN_OFFSET_V_PHASE2
#endif
#ifndef I_CT_ADC_FORMULA_OFFSET_V_PHASE3
#define I_CT_ADC_FORMULA_OFFSET_V_PHASE3 I_CT_ADC_MEAN_OFFSET_V_PHASE3
#endif

#ifndef I_OFFSET_V_PHASE1
#define I_OFFSET_V_PHASE1       0.0f
#endif
#ifndef I_OFFSET_V_PHASE2
#define I_OFFSET_V_PHASE2       0.0f
#endif
#ifndef I_OFFSET_V_PHASE3
#define I_OFFSET_V_PHASE3       0.0f
#endif

// MCP3564 is sequential-mux: V is sampled before I in each scan
// At ~1 kHz per-channel rate, V leads I by ~0.6 ms (~11° at 50 Hz)
// Subtract this from computed phase_diff. Calibrate with resistive load (cos φ=1).
#ifndef V_I_PHASE_COMP_DEG
#define V_I_PHASE_COMP_DEG    10.8f
#endif

// ── Output struct ───────────────────────────────────────────
typedef struct __attribute__((packed)) {
    // Per-phase RMS / Power
    float Vrms[N_PHASES];        // V (RMS)
    float Irms[N_PHASES];        // A (RMS)
    float P[N_PHASES];           // W   (active = mean of v*i)
    float S[N_PHASES];           // VA  (apparent = Vrms*Irms)
    float PF[N_PHASES];          // -1..1 (P/S, sign indicates direction)

    // Per-phase fundamental (50 Hz) — from Goertzel
    float amp_V[N_PHASES];       // V (peak amplitude of 50 Hz component)
    float amp_I[N_PHASES];       // A (peak amplitude of 50 Hz component)
    float angle_V[N_PHASES];     // rad (phase of V fundamental)
    float angle_I[N_PHASES];     // rad (phase of I fundamental)
    float phase_diff[N_PHASES];  // rad (V - I, compensated)

    // Per-phase THD (Total Harmonic Distortion, ratio 0..)
    float thd_V[N_PHASES];
    float thd_I[N_PHASES];

    // System-wide
    float frequency;             // Hz (from V1 zero-crossings)
    float box_temp_c;            // Protocol placeholder; Pi sends -127.0.
                                 // Temperature is measured on ESP32.

    // Meta
    uint32_t timestamp_us;       // micros() at compute time
    uint16_t n_samples;          // number of samples used
    uint16_t sample_rate_hz;     // estimated per-channel fs
} PowerMetrics;

// ── API ─────────────────────────────────────────────────────

// Compute all metrics from a flat sample block.
//   samples: [N * 6] int16, time-major (samples[t*6 + ch])
//   n_samples: number of timesteps (max 1024 — capped internally)
//   fs_per_channel: estimated samples/sec per channel
//   out: filled with results
void compute_metrics(const int16_t* samples,
                     uint32_t n_samples,
                     float fs_per_channel,
                     PowerMetrics* out);

// Serialize metrics struct to bytes (little-endian, no padding)
//   buf must be ≥ serialized_metrics_size() bytes
//   returns number of bytes written
uint16_t serialize_metrics(const PowerMetrics* m, uint8_t* buf);
uint16_t serialized_metrics_size(void);

// CRC-16-CCITT (poly 0x1021, init 0xFFFF) — used for packet framing
uint16_t crc16_ccitt(const uint8_t* data, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif // POWER_METRICS_H
