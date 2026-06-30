// ============================================================
// power_metrics.cpp
// ------------------------------------------------------------
// Implementation of electrical metric computation.
// Uses Goertzel algorithm for fundamental + harmonics — much
// lighter than full FFT when we only need specific frequency bins.
// ============================================================
#include "power_metrics.h"
#include <math.h>
#include <string.h>

static const float PI_F = 3.14159265358979323846f;

// Working buffer — sized for max window. RP2350 has 520KB SRAM,
// 1024 floats × 2 = 8 KB is trivial.
#define MAX_WORK_SAMPLES   1024
static float v_buf[MAX_WORK_SAMPLES];
static float i_buf[MAX_WORK_SAMPLES];

static const float V_SCALE[N_PHASES] = {
    V_SCALE_PHASE1, V_SCALE_PHASE2, V_SCALE_PHASE3
};

static const float I_SCALE[N_PHASES] = {
    I_SCALE_PHASE1, I_SCALE_PHASE2, I_SCALE_PHASE3
};

static const float I_OFFSET_V[N_PHASES] = {
    I_OFFSET_V_PHASE1, I_OFFSET_V_PHASE2, I_OFFSET_V_PHASE3
};

static const float I_CT_ADC_FORMULA_OFFSET_V[N_PHASES] = {
    I_CT_ADC_FORMULA_OFFSET_V_PHASE1,
    I_CT_ADC_FORMULA_OFFSET_V_PHASE2,
    I_CT_ADC_FORMULA_OFFSET_V_PHASE3
};

// Channel indices in flat buffer
static inline uint32_t V_CH(int phase) { return (uint32_t)(phase * 2); }
static inline uint32_t I_CH(int phase) { return (uint32_t)(phase * 2 + 1); }

// ============================================================
// Nonlinear CT conversion from a selected ADC statistic to current RMS.
// The chosen statistic is computed fresh for every 1-second window so the
// current estimate stays dynamic with the incoming raw waveform.
// ============================================================
static float ct_adc_stat_to_current_rms_a(float adc_stat_v, int phase) {
#if I_CT_NONLINEAR_ENABLE
    if (!isfinite(adc_stat_v) || adc_stat_v <= 0.0f || phase < 0 || phase >= N_PHASES) {
        return 0.0f;
    }

    if (adc_stat_v < I_CT_FORMULA_DEADBAND_V) {
        return 0.0f;
    }

    float formula_v = adc_stat_v - I_CT_ADC_FORMULA_OFFSET_V[phase];
    if (!isfinite(formula_v) || formula_v < I_CT_EPSILON_V) {
        return 0.0f;
    }

    float current_a = I_CT_K * powf(formula_v, I_CT_EXP);

    if (!isfinite(current_a) || current_a < 0.0f) {
        return 0.0f;
    }

    if (current_a > I_CT_MAX_A) {
        current_a = I_CT_MAX_A;
    }

    return current_a;
#else
    (void)adc_stat_v;
    (void)phase;
    return 0.0f;
#endif
}

// ============================================================
// Goertzel — single-bin DFT
//   For a real signal x[0..N-1] at sample rate fs, compute
//   the complex spectrum at target_freq.
//   Output is NOT amplitude-normalized — caller scales by 2/N
//   to get peak amplitude of a cosine at that freq.
// ============================================================
static void goertzel(const float* x, uint32_t N, float fs,
                     float target_freq,
                     float* out_real, float* out_imag) {
    float k = target_freq / fs;
    float w = 2.0f * PI_F * k;
    float cw = cosf(w);
    float sw = sinf(w);
    float c = 2.0f * cw;

    float q1 = 0.0f, q2 = 0.0f;

    for (uint32_t n = 0; n < N; ++n) {
        float q0 = c * q1 - q2 + x[n];
        q2 = q1;
        q1 = q0;
    }

    *out_real = q1 - q2 * cw;
    *out_imag = q2 * sw;
}

static float goertzel_amp(const float* x, uint32_t N, float fs, float freq) {
    float re, im;
    goertzel(x, N, fs, freq, &re, &im);
    return (2.0f * sqrtf(re * re + im * im)) / (float)N;
}

// Shift current in the same direction used by phase_diff compensation so that
// time-domain P/PF are not biased by sequential V/I acquisition.
static float compensated_active_power(const float* v, const float* i,
                                      uint32_t N, float fs) {
    float offset_samples = fs * V_I_PHASE_COMP_DEG / (360.0f * LINE_FREQ_HZ);

    if (offset_samples <= 0.0f) {
        float sum = 0.0f;

        for (uint32_t k = 0; k < N; ++k) {
            sum += v[k] * i[k];
        }

        return sum / (float)N;
    }

    uint32_t whole = (uint32_t)offset_samples;
    float frac = offset_samples - (float)whole;

    if (whole >= N) {
        return 0.0f;
    }

    float sum = 0.0f;

    if (frac <= 1.0e-6f) {
        uint32_t count = N - whole;

        for (uint32_t k = 0; k < count; ++k) {
            sum += v[k] * i[k + whole];
        }

        return sum / (float)count;
    }

    if (whole + 3U >= N) {
        return 0.0f;
    }

    float c0 = -frac * (frac - 1.0f) * (frac - 2.0f) / 6.0f;
    float c1 =  (frac + 1.0f) * (frac - 1.0f) * (frac - 2.0f) / 2.0f;
    float c2 = -(frac + 1.0f) * frac * (frac - 2.0f) / 2.0f;
    float c3 =  (frac + 1.0f) * frac * (frac - 1.0f) / 6.0f;

    uint32_t count = 0;

    for (uint32_t k = 1; k + whole + 2U < N; ++k) {
        uint32_t base = k + whole;

        float aligned_i = c0 * i[base - 1U]
                        + c1 * i[base]
                        + c2 * i[base + 1U]
                        + c3 * i[base + 2U];

        sum += v[k] * aligned_i;
        count++;
    }

    return (count > 0U) ? (sum / (float)count) : 0.0f;
}

// ============================================================
// Frequency estimation — hysteresis zero-crossing on V1.
// Plain sign-crossing is too sensitive to distorted/noisy voltage waveforms and
// can over-count, which made 50 Hz mains appear as 60-80 Hz in field tests.
// ============================================================
static float estimate_frequency(const float* x, uint32_t N, float fs) {
    if (N < 4) {
        return 0.0f;
    }

    float mean = 0.0f;

    for (uint32_t i = 0; i < N; ++i) {
        mean += x[i];
    }

    mean /= (float)N;

    float sum_sq = 0.0f;
    float peak_abs = 0.0f;

    for (uint32_t i = 0; i < N; ++i) {
        float centered = x[i] - mean;
        sum_sq += centered * centered;

        float abs_v = fabsf(centered);
        if (abs_v > peak_abs) {
            peak_abs = abs_v;
        }
    }

    float rms = sqrtf(sum_sq / (float)N);

    if (!isfinite(rms) || rms < 1.0e-3f || peak_abs < 1.0e-3f) {
        return 0.0f;
    }

    float threshold = rms * 0.25f;
    float max_threshold = peak_abs * 0.45f;

    if (threshold > max_threshold) {
        threshold = max_threshold;
    }

    if (threshold < 1.0e-3f) {
        threshold = 1.0e-3f;
    }

    float first_cross = -1.0f;
    float last_cross = -1.0f;
    uint32_t crossings = 0;
    bool armed = false;
    float prev = x[0] - mean;

    for (uint32_t i = 1; i < N; ++i) {
        float curr = x[i] - mean;

        if (curr <= -threshold) {
            armed = true;
        }

        if (armed && prev < threshold && curr >= threshold) {
            float denom = curr - prev;
            float frac = (fabsf(denom) > 1.0e-9f)
                       ? ((threshold - prev) / denom)
                       : 0.0f;

            if (frac < 0.0f) {
                frac = 0.0f;
            }

            if (frac > 1.0f) {
                frac = 1.0f;
            }

            float cross = ((float)i - 1.0f) + frac;

            if (first_cross < 0.0f) {
                first_cross = cross;
            }

            last_cross = cross;
            crossings++;
            armed = false;
        }

        prev = curr;
    }

    if (crossings < 2 || first_cross < 0.0f || last_cross <= first_cross) {
        return 0.0f;
    }

    float dt_s = (last_cross - first_cross) / fs;

    if (dt_s <= 0.0f) {
        return 0.0f;
    }

    return (float)(crossings - 1) / dt_s;
}

// ============================================================
// compute_metrics — main entry
// ============================================================
void compute_metrics(const int16_t* samples,
                     uint32_t n_samples,
                     float fs_per_channel,
                     PowerMetrics* m) {
    memset(m, 0, sizeof(*m));

    uint32_t N = n_samples;

    if (N > MAX_WORK_SAMPLES) {
        N = MAX_WORK_SAMPLES;
    }

    if (N < 2) {
        return;
    }

    m->n_samples = (uint16_t)N;
    m->sample_rate_hz = (uint16_t)fs_per_channel;

    if (fs_per_channel < 1.0f) {
        return;  // protect divide-by-zero
    }

    for (int p = 0; p < N_PHASES; ++p) {
        const float vsc = V_SCALE[p];
        const float isc = I_SCALE[p];

        // 1) Convert int16 → float, then measure mean for DC removal
        float v_mean = 0.0f;
        float i_mean = 0.0f;

#if I_CT_NONLINEAR_ENABLE
        float i_adc_sum_v = 0.0f;
        (void)isc;
#endif

        for (uint32_t k = 0; k < N; ++k) {
            float vv = (float)samples[k * 6 + V_CH(p)] * vsc;
            v_buf[k] = vv;
            v_mean += vv;

#if I_CT_NONLINEAR_ENABLE
            // Current raw int16 is stored as ADC-side volts * 5000.
            // So reconstruct ADC voltage first:
            //   V_adc = raw_I / 5000.0
            float i_adc_v = (float)samples[k * 6 + I_CH(p)] / 5000.0f;
            i_buf[k] = i_adc_v;
            i_adc_sum_v += i_adc_v;
#else
            // Legacy linear current conversion
            float ii = (float)samples[k * 6 + I_CH(p)] * isc;
            i_buf[k] = ii;
            i_mean += ii;
#endif
        }

        v_mean /= (float)N;

#if I_CT_NONLINEAR_ENABLE
        float i_adc_mean_v = i_adc_sum_v / (float)N;
        float adc_sum_sq = 0.0f;
        float adc_mean_abs_v = 0.0f;
        float adc_peak_abs_v = 0.0f;

        for (uint32_t k = 0; k < N; ++k) {
            float centered_adc_v = i_buf[k] - i_adc_mean_v;
            float abs_centered_adc_v = fabsf(centered_adc_v);
            i_buf[k] = centered_adc_v;
            adc_sum_sq += centered_adc_v * centered_adc_v;
            adc_mean_abs_v += abs_centered_adc_v;
            if (abs_centered_adc_v > adc_peak_abs_v) {
                adc_peak_abs_v = abs_centered_adc_v;
            }
        }

        float adc_rms_v = sqrtf(adc_sum_sq / (float)N);
        adc_mean_abs_v /= (float)N;

        float adc_formula_v = adc_mean_abs_v;
#if I_CT_STAT_MODE == I_CT_STAT_MODE_RMS
        adc_formula_v = adc_rms_v;
#elif I_CT_STAT_MODE == I_CT_STAT_MODE_PEAK
        adc_formula_v = adc_peak_abs_v / I_CT_PEAK_TO_FORMULA_DIVISOR;
#endif

        float target_Irms = ct_adc_stat_to_current_rms_a(adc_formula_v, p);
        float current_scale = (adc_rms_v > 1.0e-9f) ? (target_Irms / adc_rms_v) : 0.0f;

        for (uint32_t k = 0; k < N; ++k) {
            float ii = i_buf[k] * current_scale;
            i_buf[k] = ii;
            i_mean += ii;
        }
#else
        (void)isc;
#endif

        i_mean /= (float)N;

        // 2) DC removal in place
        for (uint32_t k = 0; k < N; ++k) {
            v_buf[k] -= v_mean;
            i_buf[k] -= i_mean;
        }

        // 3) Time-domain: Vrms, Irms, P, S, PF
        float sum_v2 = 0.0f;
        float sum_i2 = 0.0f;

        for (uint32_t k = 0; k < N; ++k) {
            sum_v2 += v_buf[k] * v_buf[k];
            sum_i2 += i_buf[k] * i_buf[k];
        }

        float Vrms = sqrtf(sum_v2 / (float)N);
        float Irms = sqrtf(sum_i2 / (float)N);
        float P = compensated_active_power(v_buf, i_buf, N, fs_per_channel);
        float S = Vrms * Irms;
        float PF = (S > 1e-3f) ? (P / S) : 0.0f;

        if (PF > 1.0f) {
            PF = 1.0f;
        }

        if (PF < -1.0f) {
            PF = -1.0f;
        }

        m->Vrms[p] = Vrms;
        m->Irms[p] = Irms;
        m->P[p]    = P;
        m->S[p]    = S;
        m->PF[p]   = PF;

        // 4) Fundamental 50 Hz — Goertzel on V and I
        float vr, vi, ir, ii;

        goertzel(v_buf, N, fs_per_channel, LINE_FREQ_HZ, &vr, &vi);
        goertzel(i_buf, N, fs_per_channel, LINE_FREQ_HZ, &ir, &ii);

        float v_mag = sqrtf(vr * vr + vi * vi);
        float i_mag = sqrtf(ir * ir + ii * ii);

        float amp_V = (2.0f * v_mag) / (float)N;
        float amp_I = (2.0f * i_mag) / (float)N;

        float ang_V = atan2f(vi, vr);
        float ang_I = atan2f(ii, ir);

        // Phase diff with sequential-mux compensation
        float pd = ang_V - ang_I;
        pd -= V_I_PHASE_COMP_DEG * (PI_F / 180.0f);

        // Wrap to [-pi, pi]
        while (pd > PI_F) {
            pd -= 2.0f * PI_F;
        }

        while (pd < -PI_F) {
            pd += 2.0f * PI_F;
        }

        m->amp_V[p]      = amp_V;
        m->amp_I[p]      = amp_I;
        m->angle_V[p]    = ang_V;
        m->angle_I[p]    = ang_I;
        m->phase_diff[p] = pd;

        // 5) Harmonics: THD = sqrt(sum H_k^2 for k=2..7) / H_1
        float h_v_sq_sum = 0.0f;
        float h_i_sq_sum = 0.0f;

        for (int h = 2; h <= N_HARMONICS; ++h) {
            float fh = LINE_FREQ_HZ * (float)h;

            if (fh > fs_per_channel * 0.5f) {
                break;  // Nyquist
            }

            float hv = goertzel_amp(v_buf, N, fs_per_channel, fh);
            float hi = goertzel_amp(i_buf, N, fs_per_channel, fh);

            h_v_sq_sum += hv * hv;
            h_i_sq_sum += hi * hi;
        }

        m->thd_V[p] = (amp_V > 1e-3f) ? (sqrtf(h_v_sq_sum) / amp_V) : 0.0f;
        m->thd_I[p] = (amp_I > 1e-3f) ? (sqrtf(h_i_sq_sum) / amp_I) : 0.0f;
    }

    // 6) Frequency — re-fill v_buf with V1
    // v_buf was holding the last phase after loop, so load V1 again.
    for (uint32_t k = 0; k < N; ++k) {
        v_buf[k] = (float)samples[k * 6 + V_CH(0)] * V_SCALE[0];
    }

    float measured_frequency = estimate_frequency(v_buf, N, fs_per_channel);

    if (!isfinite(measured_frequency) ||
        measured_frequency < 45.0f ||
        measured_frequency > 55.0f) {
        measured_frequency = LINE_FREQ_HZ;
    }

    m->frequency = measured_frequency;
}

// ============================================================
// Serialization (little-endian, packed)
// ============================================================
static uint8_t* w_f32(uint8_t* p, float v) {
    memcpy(p, &v, 4);
    return p + 4;
}

static uint8_t* w_u32(uint8_t* p, uint32_t v) {
    memcpy(p, &v, 4);
    return p + 4;
}

static uint8_t* w_u16(uint8_t* p, uint16_t v) {
    memcpy(p, &v, 2);
    return p + 2;
}

uint16_t serialized_metrics_size(void) {
    // 12 × 3 floats (per-phase fields) + 2 system floats (freq, box_temp)
    // + 1 uint32 + 2 uint16
    // = (36 + 2) * 4 + 4 + 2 + 2 = 160 bytes
    return 160;
}

uint16_t serialize_metrics(const PowerMetrics* m, uint8_t* buf) {
    uint8_t* p = buf;

    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->Vrms[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->Irms[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->P[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->S[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->PF[i]);

    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->amp_V[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->amp_I[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->angle_V[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->angle_I[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->phase_diff[i]);

    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->thd_V[i]);
    for (int i = 0; i < N_PHASES; ++i) p = w_f32(p, m->thd_I[i]);

    p = w_f32(p, m->frequency);
    p = w_f32(p, m->box_temp_c);
    p = w_u32(p, m->timestamp_us);
    p = w_u16(p, m->n_samples);
    p = w_u16(p, m->sample_rate_hz);

    return (uint16_t)(p - buf);
}

// ============================================================
// CRC-16-CCITT (poly 0x1021, init 0xFFFF, no reflection)
// ============================================================
uint16_t crc16_ccitt(const uint8_t* data, uint32_t len) {
    uint16_t crc = 0xFFFF;

    for (uint32_t i = 0; i < len; ++i) {
        crc ^= ((uint16_t)data[i]) << 8;

        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000) {
                crc = (uint16_t)((crc << 1) ^ 0x1021);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }

    return crc;
}
